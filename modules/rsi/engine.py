#!/usr/bin/env python
"""
Recursive Self-Improvement (RSI) Engine for SAAF-OS

This module implements the Recursive Self-Improvement (RSI) Engine, which is responsible for:
1. Generating and proposing patches to the system
2. Handling governance gates via the message bus
3. Managing patch rollbacks if necessary
"""

import os
import json
import uuid
import logging
import random
import string
import time
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from .crypto_utils import sign_patch, verify_patch_signature

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RSIEngine")


class Patch:
    """
    Represents a system patch that modifies some component of the SAAF-OS.
    """
    
    def __init__(self, 
                 module_path: str, 
                 content: str, 
                 description: str,
                 author: str = "RSI-Engine",
                 patch_id: Optional[str] = None,
                 signatures: Optional[list] = None):
        """
        Initialize a patch.
        
        Args:
            module_path: Path to the module being modified
            content: The modified content (could be a diff or full file)
            description: Description of what the patch does
            author: Who/what created the patch
            patch_id: Unique identifier for the patch (generated if None)
            signatures: List of signatures for the patch
        """
        self.module_path = module_path
        self.content = content
        self.description = description
        self.author = author
        self.creation_time = time.time()
        self.patch_id = patch_id if patch_id else str(uuid.uuid4())
        self.status = "proposed"  # proposed, approved, rejected, applied, rolled_back
        self.signatures = signatures or []  # List of dicts: [{signer, signature}]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert patch to a dictionary.
        
        Returns:
            Dictionary representation of the patch
        """
        return {
            "patch_id": self.patch_id,
            "module_path": self.module_path,
            "description": self.description,
            "author": self.author,
            "creation_time": self.creation_time,
            "status": self.status,
            "content": self.content,
            "signatures": self.signatures
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Patch':
        """
        Create a patch from a dictionary.
        
        Args:
            data: Dictionary representation of a patch
            
        Returns:
            Patch object
        """
        patch = cls(
            module_path=data["module_path"],
            content=data["content"],
            description=data["description"],
            author=data["author"],
            patch_id=data["patch_id"],
            signatures=data.get("signatures", [])
        )
        patch.creation_time = data["creation_time"]
        patch.status = data["status"]
        return patch


class RSIEngine:
    """
    Main class implementing the Recursive Self-Improvement Engine.
    """
    
    def __init__(self, message_bus=None):
        """
        Initialize the RSI Engine.
        
        Args:
            message_bus: Message bus adapter for communication
        """
        self.message_bus = message_bus
        self.patches: Dict[str, Patch] = {}  # Maps patch_id to Patch
        self.rollback_table: Dict[str, str] = {}  # Maps patch_id to original content
        self.active_patches: List[str] = []  # List of applied patch IDs
        
        # Register message handlers if message bus is provided
        if self.message_bus:
            self.register_handlers()
    
    def register_handlers(self) -> None:
        """Register handlers for relevant message types on the message bus."""
        self.message_bus.subscribe("rsi.patch.governance_decision", self._handle_governance_decision)
        self.message_bus.subscribe("rsi.patch.evaluate", self._handle_patch_evaluation)
    
    def generate_patch(self, module_path: str, description: str = "", critical: bool = False) -> Patch:
        """
        Generate a simple random patch for demonstration purposes.
        In a real implementation, this would involve more sophisticated analysis.
        
        Args:
            module_path: Path to the module to patch
            description: Description of what the patch aims to do
            critical: Whether the patch is critical and requires two signatures
            
        Returns:
            Generated patch
        """
        # In the MVP, we just generate a random diff as a placeholder
        # In a real implementation, this would analyze code and make meaningful changes
        random_str = ''.join(random.choice(string.ascii_letters) for _ in range(20))
        content = f"@@ -1,3 +1,3 @@\n-old line\n+{random_str}\n context"
        
        patch = Patch(
            module_path=module_path,
            content=content,
            description=description or f"Auto-generated patch for {module_path}"
        )
        
        # Sign patch
        patch_bytes = json.dumps({k: v for k, v in patch.to_dict().items() if k != 'signatures'}, sort_keys=True).encode()
        sig = sign_patch(patch_bytes, key_name="rsi_signer")
        patch.signatures.append({"signer": "rsi_signer", "signature": sig})
        
        # For critical patches, require a second signature (simulate for now)
        if critical:
            sig2 = sign_patch(patch_bytes, key_name="rsi_signer2") if os.path.exists(os.path.join(os.path.dirname(__file__), '../../data/keys/rsi_signer2_private.key')) else sig
            patch.signatures.append({"signer": "rsi_signer2", "signature": sig2})
        
        self.patches[patch.patch_id] = patch
        return patch
    
    def propose_patch(self, patch: Patch) -> bool:
        """
        Propose a patch to the governance system via the message bus.
        
        Args:
            patch: The patch to propose
            
        Returns:
            True if successfully proposed, False otherwise
        """
        if not self.message_bus:
            logger.error("Cannot propose patch: no message bus connected")
            return False
        
        logger.info(f"Proposing patch {patch.patch_id}: {patch.description}")
        
        # Store the patch
        self.patches[patch.patch_id] = patch
        
        # Publish to the message bus
        self.message_bus.publish("rsi.patch.proposed", {
            "patch": patch.to_dict()
        })
        
        return True
    
    def verify_patch_signatures(self, patch: Patch, critical: bool = False) -> bool:
        """
        Verify the signatures of a patch.
        
        Args:
            patch: The patch to verify
            critical: Whether the patch is critical and requires two signatures
            
        Returns:
            True if signatures are valid, False otherwise
        """
        patch_bytes = json.dumps({k: v for k, v in patch.to_dict().items() if k != 'signatures'}, sort_keys=True).encode()
        if critical:
            if len(patch.signatures) < 2:
                return False
            return all(
                verify_patch_signature(patch_bytes, s["signature"], key_name=s["signer"]) for s in patch.signatures[:2]
            )
        else:
            if not patch.signatures:
                return False
            return verify_patch_signature(patch_bytes, patch.signatures[0]["signature"], key_name=patch.signatures[0]["signer"])
    
    def apply_patch(self, patch_id: str, critical: bool = False) -> bool:
        """
        Apply a patch to the system.
        
        Args:
            patch_id: ID of the patch to apply
            critical: Whether the patch is critical and requires two signatures
            
        Returns:
            True if successfully applied, False otherwise
        """
        if patch_id not in self.patches:
            logger.error(f"Cannot apply patch: {patch_id} not found")
            return False
        
        patch = self.patches[patch_id]
        
        if not self.verify_patch_signatures(patch, critical=critical):
            logger.error(f"Patch {patch_id} signature verification failed. Not applying.")
            return False
        
        logger.info(f"Applying patch {patch_id}: {patch.description}")
        
        # In a real implementation, we would:
        # 1. Back up the original file content
        # 2. Apply the diff to the file
        # 3. Update the system
        
        # For MVP, we just simulate these steps
        try:
            # Get original content (mock)
            original_content = f"Original content of {patch.module_path}"
            
            # Store in rollback table
            self.rollback_table[patch_id] = original_content
            
            # Mark as applied
            patch.status = "applied"
            self.active_patches.append(patch_id)
            
            # Notify via message bus
            if self.message_bus:
                self.message_bus.publish("rsi.patch.applied", {
                    "patch_id": patch_id,
                    "success": True
                })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply patch {patch_id}: {str(e)}")
            return False
    
    def rollback_patch(self, patch_id: str) -> bool:
        """
        Roll back a previously applied patch.
        
        Args:
            patch_id: ID of the patch to roll back
            
        Returns:
            True if successfully rolled back, False otherwise
        """
        if patch_id not in self.patches or patch_id not in self.rollback_table:
            logger.error(f"Cannot rollback patch: {patch_id} not found or not applied")
            return False
        
        patch = self.patches[patch_id]
        logger.info(f"Rolling back patch {patch_id}: {patch.description}")
        
        # In a real implementation, we would:
        # 1. Restore the original file content
        # 2. Update the system
        
        # For MVP, we just simulate these steps
        try:
            # Mark as rolled back
            patch.status = "rolled_back"
            if patch_id in self.active_patches:
                self.active_patches.remove(patch_id)
            
            # Notify via message bus
            if self.message_bus:
                self.message_bus.publish("rsi.patch.rolled_back", {
                    "patch_id": patch_id,
                    "success": True
                })
            
            logger.info(f"RSI patch vetoed by governance → rollback successful")
            return True
            
        except Exception as e:
            logger.error(f"Failed to roll back patch {patch_id}: {str(e)}")
            return False
    
    def _handle_governance_decision(self, message: Dict[str, Any]) -> None:
        """
        Handle governance decision messages from the bus.
        
        Args:
            message: Message containing governance decision
        """
        if "patch_id" not in message or "decision" not in message:
            logger.error("Invalid governance decision message")
            return
        
        patch_id = message["patch_id"]
        decision = message["decision"]
        
        if patch_id not in self.patches:
            logger.error(f"Unknown patch ID in governance decision: {patch_id}")
            return
        
        patch = self.patches[patch_id]
        
        if decision == "approved":
            logger.info(f"Patch {patch_id} approved by governance")
            patch.status = "approved"
            self.apply_patch(patch_id)
        elif decision == "rejected":
            logger.info(f"Patch {patch_id} rejected by governance")
            patch.status = "rejected"
        elif decision == "veto":
            logger.info(f"Patch {patch_id} vetoed by governance")
            if patch.status == "applied":
                self.rollback_patch(patch_id)
            else:
                patch.status = "rejected"
        else:
            logger.warning(f"Unknown decision for patch {patch_id}: {decision}")
    
    def _handle_patch_evaluation(self, message: Dict[str, Any]) -> None:
        """
        Handle patch evaluation requests from the bus.
        
        Args:
            message: Message containing evaluation request
        """
        if "patch_id" not in message:
            logger.error("Invalid patch evaluation message")
            return
        
        patch_id = message["patch_id"]
        
        if patch_id not in self.patches:
            logger.error(f"Unknown patch ID in evaluation request: {patch_id}")
            return
        
        # In a real implementation, this would run tests, validate the patch,
        # check for security issues, etc.
        
        # For MVP, we just respond with a random quality score
        quality_score = random.random()
        compatibility_score = random.random()
        security_score = random.random()
        
        # Respond with evaluation results
        if self.message_bus:
            self.message_bus.publish("rsi.patch.evaluation_result", {
                "patch_id": patch_id,
                "quality_score": quality_score,
                "compatibility_score": compatibility_score,
                "security_score": security_score,
                "recommendation": "approve" if all(s > 0.5 for s in [quality_score, compatibility_score, security_score]) else "reject"
            })


class DummyRSIEngine(RSIEngine):
    """
    Dummy implementation of RSI Engine for testing.
    """
    
    def __init__(self, message_bus=None):
        """Initialize the dummy RSI engine."""
        super().__init__(message_bus)
    
    def generate_patch(self, module_path: str, description: str = "") -> Patch:
        """Generate a deterministic patch for testing."""
        content = f"@@ -1,3 +1,3 @@\n-old line\n+TEST_PATCH\n context"
        
        patch = Patch(
            module_path=module_path,
            content=content,
            description=description or f"Test patch for {module_path}",
            patch_id="test_patch_" + str(uuid.uuid4())[:8]
        )
        
        self.patches[patch.patch_id] = patch
        return patch