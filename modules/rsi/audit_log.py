#!/usr/bin/env python
"""
Audit logging system for RSI governance events.

This module provides tamper-evident logging for all RSI-related events,
including patch proposals, votes, approvals, vetoes, and rollbacks.
"""

import os
import json
import time
import logging
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime
import hmac

from .crypto_utils import sign_patch, verify_patch_signature

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RSIAuditLog")

# Constants
AUDIT_LOG_DIR = os.path.expanduser("~/.saaf_os/audit_logs")
CHAIN_FILE = os.path.join(AUDIT_LOG_DIR, "rsi_event_chain.json")
EVENT_TYPES = [
    "proposal", "vote_result", "patch_applied", "patch_rolled_back", 
    "tension_spike", "veto", "governance_decision"
]


class AuditLog:
    """
    Tamper-evident audit log for RSI governance events.
    
    Each event is cryptographically linked to the previous events,
    forming a verifiable chain that cannot be modified without detection.
    """
    
    def __init__(self, in_memory=False):
        """
        Initialize the audit log system.
        
        Args:
            in_memory: If True, don't persist logs to disk (for testing)
        """
        self.events = []
        self.in_memory = in_memory
        
        if not in_memory:
            self._ensure_log_directory()
            self.load_events()
    
    def _ensure_log_directory(self):
        """Create the log directory if it doesn't exist."""
        os.makedirs(AUDIT_LOG_DIR, exist_ok=True)
    
    def clear(self):
        """Clear all events from the log (mainly for testing)."""
        self.events = []
        if not self.in_memory:
            if os.path.exists(CHAIN_FILE):
                os.remove(CHAIN_FILE)
    
    def load_events(self):
        """Load the existing event chain from disk."""
        if self.in_memory or not os.path.exists(CHAIN_FILE):
            self.events = []
            return
        
        try:
            with open(CHAIN_FILE, 'r') as f:
                self.events = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning("Failed to load existing audit log, starting fresh")
            self.events = []
    
    def _save_events(self):
        """Save the event chain to disk."""
        if self.in_memory:
            return
            
        with open(CHAIN_FILE, 'w') as f:
            json.dump(self.events, f, indent=2)
    
    def _compute_event_hash(self, event: Dict[str, Any]) -> str:
        """
        Compute a deterministic hash of an event, excluding the hash field itself.
        
        Args:
            event: The event to hash
            
        Returns:
            Hash of the event as a hex string
        """
        event_copy = event.copy()
        # Remove fields that shouldn't be part of the hash
        event_copy.pop("hash", None)
        event_copy.pop("signature", None)
        
        # Create a deterministic string representation
        serialized = json.dumps(event_copy, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    
    def _sign_event(self, event: Dict[str, Any], key_name: str = "rsi_signer") -> str:
        """
        Sign an event using the RSI key.
        
        Args:
            event: The event to sign
            key_name: Name of the key to use for signing
            
        Returns:
            Signature as a hex string
        """
        # Hash the event first
        event_hash = event["hash"]
        
        # Use the same signing mechanism as for patches
        # We're signing the hash, not the full event
        message = {"hash": event_hash}
        signature = sign_patch(message, key_name)
        return signature
    
    def log_event(self, event_type: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Append a new event to the audit log.
        
        Args:
            event_type: Type of event, must be one of EVENT_TYPES
            metadata: Additional event data
            
        Returns:
            The created event record
        """
        if event_type not in EVENT_TYPES:
            raise ValueError(f"Invalid event type: {event_type}. Must be one of {EVENT_TYPES}")
        
        # Create the event record
        event = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "unix_time": time.time(),
            "metadata": metadata,
        }
        
        # Add chain link to previous event if any
        if self.events:
            event["previous_hash"] = self.events[-1]["hash"]
        
        # Compute hash and sign
        event["hash"] = self._compute_event_hash(event)
        event["signature"] = self._sign_event(event)
        
        # Append and save
        self.events.append(event)
        self._save_events()
        
        logger.info(f"Logged RSI event: {event_type}")
        return event
    
    def verify_chain_integrity(self) -> bool:
        """
        Verify the integrity of the entire event chain.
        
        Returns:
            True if the chain is valid, False otherwise
        """
        if not self.events:
            return True  # Empty chain is valid
        
        for i, event in enumerate(self.events):
            # Verify hash
            computed_hash = self._compute_event_hash(event)
            if computed_hash != event["hash"]:
                logger.error(f"Hash mismatch at event {i}")
                return False
            
            # Verify signature
            message = {"hash": event["hash"]}
            if not verify_patch_signature(message, event["signature"]):
                logger.error(f"Signature verification failed at event {i}")
                return False
            
            # Verify chain link (except for first event)
            if i > 0:
                previous_hash = self.events[i-1]["hash"]
                if event.get("previous_hash") != previous_hash:
                    logger.error(f"Chain link broken at event {i}")
                    return False
        
        return True
    
    def get_events(self, since: Optional[datetime] = None, 
                  event_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve events from the audit log, optionally filtered by time and type.
        
        Args:
            since: If provided, only returns events after this time
            event_types: If provided, only returns events of these types
            
        Returns:
            List of matching events
        """
        filtered_events = self.events
        
        # Filter by timestamp if needed
        if since:
            since_iso = since.isoformat()
            filtered_events = [e for e in filtered_events if e["timestamp"] >= since_iso]
        
        # Filter by event type if needed
        if event_types:
            filtered_events = [e for e in filtered_events if e["event_type"] in event_types]
            
        return filtered_events


# Singleton pattern for global access
_audit_log_instance = None

def get_audit_log(in_memory=False) -> AuditLog:
    """
    Get the singleton audit log instance.
    
    Args:
        in_memory: If True, use an in-memory log (for testing)
        
    Returns:
        AuditLog instance
    """
    global _audit_log_instance
    if _audit_log_instance is None or in_memory:
        _audit_log_instance = AuditLog(in_memory=in_memory)
    return _audit_log_instance


# Convenience functions
def log_rsi_event(event_type: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Append structured governance-relevant event to tamper-evident audit log.
    Each event is time-stamped and cryptographically signed.
    
    Args:
        event_type: Type of event, must be one of EVENT_TYPES
        metadata: Additional event data
        
    Returns:
        The created event record
    """
    audit_log = get_audit_log()
    return audit_log.log_event(event_type, metadata)

def get_rsi_audit_log(since: Optional[datetime] = None, 
                     event_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Query recent governance events for review.
    
    Args:
        since: If provided, only returns events after this time
        event_types: If provided, only returns events of these types
        
    Returns:
        List of matching events, e.g. 
        [{"event": "veto", "patch_id": "abc123", "reason": "...", "timestamp": ...}]
    """
    audit_log = get_audit_log()
    return audit_log.get_events(since, event_types)