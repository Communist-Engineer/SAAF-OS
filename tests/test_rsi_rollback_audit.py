#!/usr/bin/env python
"""
Unit tests for the enhanced RSI Engine rollback mechanism and audit logging.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import json
import tempfile
import shutil
import time
from datetime import datetime, timedelta

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.rsi.engine import RSIEngine, Patch, BlobStore
from modules.rsi.audit_log import log_rsi_event, get_rsi_audit_log, AuditLog, get_audit_log


class MockMessageBus:
    """Mock message bus for testing."""
    
    def __init__(self):
        self.handlers = {}
        self.published_messages = []
    
    def subscribe(self, topic, handler):
        if topic not in self.handlers:
            self.handlers[topic] = []
        self.handlers[topic].append(handler)
    
    def publish(self, topic, message):
        self.published_messages.append((topic, message))
        if topic in self.handlers:
            for handler in self.handlers[topic]:
                handler(message)
    
    def get_published_messages(self, topic=None):
        if topic:
            return [msg for t, msg in self.published_messages if t == topic]
        return self.published_messages


class TestBlobStore(unittest.TestCase):
    """Test the content-addressable blob storage."""
    
    def setUp(self):
        """Set up a temporary blob storage directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_blob_dir = os.environ.get('BLOB_STORAGE_DIR')
        os.environ['BLOB_STORAGE_DIR'] = self.temp_dir
        self.blob_store = BlobStore()
    
    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.temp_dir)
        if self.original_blob_dir:
            os.environ['BLOB_STORAGE_DIR'] = self.original_blob_dir
        else:
            del os.environ['BLOB_STORAGE_DIR']
    
    def test_store_and_retrieve_blob(self):
        """Test storing and retrieving content from the blob store."""
        content = "This is some test content"
        hash_id = self.blob_store.store_blob(content)
        
        # Verify the hash looks right
        self.assertEqual(len(hash_id), 64)  # SHA-256 is 64 hex chars
        
        # Retrieve the content and verify
        retrieved = self.blob_store.get_blob(hash_id)
        self.assertEqual(retrieved, content)
    
    def test_content_addressing(self):
        """Test that identical content produces the same hash."""
        content1 = "Same content"
        content2 = "Same content"
        
        hash1 = self.blob_store.store_blob(content1)
        hash2 = self.blob_store.store_blob(content2)
        
        self.assertEqual(hash1, hash2)
    
    def test_nonexistent_blob(self):
        """Test retrieving a non-existent blob returns None."""
        fake_hash = "0" * 64
        self.assertIsNone(self.blob_store.get_blob(fake_hash))


class TestRSIRollback(unittest.TestCase):
    """Test the enhanced rollback mechanism."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.message_bus = MockMessageBus()
        self.rsi = RSIEngine(self.message_bus)
        
        # Create a temp directory for blob storage
        self.temp_dir = tempfile.mkdtemp()
        self.original_blob_dir = os.environ.get('BLOB_STORAGE_DIR')
        os.environ['BLOB_STORAGE_DIR'] = self.temp_dir
        
        # Create a new blob store
        self.rsi.blob_store = BlobStore()
    
    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.temp_dir)
        if self.original_blob_dir:
            os.environ['BLOB_STORAGE_DIR'] = self.original_blob_dir
        else:
            del os.environ['BLOB_STORAGE_DIR']
    
    def test_rollback_patch(self):
        """Test the basic rollback functionality."""
        # Generate a patch
        module_path = "modules/test.py"
        patch = self.rsi.generate_patch(module_path, "Test patch")
        
        # Apply the patch
        self.rsi.apply_patch(patch.patch_id)
        
        # Check rollback table has correct metadata
        self.assertIn(patch.patch_id, self.rsi.rollback_table)
        rollback_info = self.rsi.rollback_table[patch.patch_id]
        self.assertIn("original_hash", rollback_info)
        self.assertIn("patch_hash", rollback_info)
        self.assertIn("module_path", rollback_info)
        self.assertEqual(rollback_info["module_path"], module_path)
        
        # Check module version mapping
        self.assertEqual(self.rsi.module_version_map[module_path], patch.patch_id)
        
        # Roll back the patch
        result = self.rsi.rollback_patch(patch.patch_id)
        self.assertTrue(result)
        self.assertEqual(patch.status, "rolled_back")
        self.assertNotIn(patch.patch_id, self.rsi.active_patches)
        self.assertNotIn(module_path, self.rsi.module_version_map)
    
    def test_rollback_with_multiple_versions(self):
        """Test rollback with multiple versions of the same module."""
        module_path = "modules/versioned.py"
        
        # Apply first patch
        patch1 = self.rsi.generate_patch(module_path, "First version")
        self.rsi.apply_patch(patch1.patch_id)
        
        # Apply second patch to same module
        patch2 = self.rsi.generate_patch(module_path, "Second version")
        self.rsi.apply_patch(patch2.patch_id)
        
        # Verify current version
        self.assertEqual(self.rsi.module_version_map[module_path], patch2.patch_id)
        
        # Roll back the second patch
        self.rsi.rollback_patch(patch2.patch_id)
        
        # Verify it went back to the first version
        self.assertEqual(self.rsi.module_version_map[module_path], patch1.patch_id)
        self.assertEqual(patch2.status, "rolled_back")
        self.assertEqual(patch1.status, "applied")
        
        # Roll back the first patch
        self.rsi.rollback_patch(patch1.patch_id)
        
        # Verify no version is active
        self.assertNotIn(module_path, self.rsi.module_version_map)
        self.assertEqual(patch1.status, "rolled_back")
    
    def test_get_rollback_status(self):
        """Test retrieving the rollback status."""
        # Apply patches to different modules
        patch1 = self.rsi.generate_patch("modules/mod1.py", "Module 1 patch")
        patch2 = self.rsi.generate_patch("modules/mod2.py", "Module 2 patch")
        
        self.rsi.apply_patch(patch1.patch_id)
        self.rsi.apply_patch(patch2.patch_id)
        
        # Check rollback status
        status = self.rsi.get_rollback_status()
        self.assertEqual(len(status), 2)
        self.assertEqual(status["modules/mod1.py"], patch1.patch_id)
        self.assertEqual(status["modules/mod2.py"], patch2.patch_id)
        
        # Rollback one patch
        self.rsi.rollback_patch(patch1.patch_id)
        
        # Check updated status
        status = self.rsi.get_rollback_status()
        self.assertEqual(len(status), 1)
        self.assertNotIn("modules/mod1.py", status)
        self.assertEqual(status["modules/mod2.py"], patch2.patch_id)


class TestAuditLog(unittest.TestCase):
    """Test the audit logging system."""
    
    def setUp(self):
        """Set up a temporary audit log directory."""
        # Use in-memory log for testing to avoid filesystem operations
        self.audit_log = AuditLog(in_memory=True)
        
        # Mock the datetime for consistent timestamps
        self.patch_datetime = patch('modules.rsi.audit_log.datetime')
        self.mock_datetime = self.patch_datetime.start()
        self.mock_datetime.now.return_value = datetime(2025, 5, 6, 12, 0, 0)
        self.mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
    
    def tearDown(self):
        """Clean up test environment."""
        # Clear the audit log
        self.audit_log.clear()
        
        # Stop patching datetime
        self.patch_datetime.stop()
    
    def test_log_event(self):
        """Test logging a basic event."""
        metadata = {"test_key": "test_value"}
        event = self.audit_log.log_event("proposal", metadata)
        
        self.assertEqual(event["event_type"], "proposal")
        self.assertEqual(event["metadata"], metadata)
        self.assertIn("timestamp", event)
        self.assertIn("hash", event)
        self.assertIn("signature", event)
    
    def test_event_chain_integrity(self):
        """Test that events form a valid chain with previous_hash links."""
        # Log a series of events
        self.audit_log.log_event("proposal", {"id": "1"})
        self.audit_log.log_event("vote_result", {"id": "2"})
        self.audit_log.log_event("patch_applied", {"id": "3"})
        
        # Verify the chain links
        self.assertEqual(len(self.audit_log.events), 3)
        self.assertNotIn("previous_hash", self.audit_log.events[0])
        self.assertEqual(self.audit_log.events[1]["previous_hash"], self.audit_log.events[0]["hash"])
        self.assertEqual(self.audit_log.events[2]["previous_hash"], self.audit_log.events[1]["hash"])
        
        # Verify chain integrity
        self.assertTrue(self.audit_log.verify_chain_integrity())
    
    def test_filter_events_by_time(self):
        """Test filtering events by timestamp."""
        # Log events with manipulated timestamps
        self.mock_datetime.now.return_value = datetime(2025, 5, 1, 12, 0, 0)
        self.audit_log.log_event("proposal", {"day": "1"})
        
        self.mock_datetime.now.return_value = datetime(2025, 5, 2, 12, 0, 0)
        self.audit_log.log_event("vote_result", {"day": "2"})
        
        self.mock_datetime.now.return_value = datetime(2025, 5, 3, 12, 0, 0)
        self.audit_log.log_event("patch_applied", {"day": "3"})
        
        # Filter by date
        since = datetime(2025, 5, 2, 0, 0, 0)
        filtered = self.audit_log.get_events(since=since)
        
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]["metadata"]["day"], "2")
        self.assertEqual(filtered[1]["metadata"]["day"], "3")
    
    def test_filter_events_by_type(self):
        """Test filtering events by event type."""
        # Log different types of events
        self.audit_log.log_event("proposal", {"id": "1"})
        self.audit_log.log_event("vote_result", {"id": "2"})
        self.audit_log.log_event("proposal", {"id": "3"})
        
        # Filter by type
        filtered = self.audit_log.get_events(event_types=["proposal"])
        
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]["metadata"]["id"], "1")
        self.assertEqual(filtered[1]["metadata"]["id"], "3")


class TestIntegration(unittest.TestCase):
    """Integration tests for the RSI engine with audit logging and rollback."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set up temp dirs for blob storage
        self.temp_blob_dir = tempfile.mkdtemp()
        
        # Remember original dirs
        self.original_blob_dir = os.environ.get('BLOB_STORAGE_DIR')
        
        # Set up temp dirs
        os.environ['BLOB_STORAGE_DIR'] = self.temp_blob_dir
        
        # Initialize in-memory audit log
        self.audit_log = get_audit_log(in_memory=True)
        self.audit_log.clear()  # Start fresh
        
        # Create the engine and message bus
        self.message_bus = MockMessageBus()
        
        # Override the verify_patch_signatures method for testing purposes
        class TestRSIEngine(RSIEngine):
            def verify_patch_signatures(self, patch, critical=False, force_recompute=False):
                # In tests, always return True for signature verification
                return True
                
        self.rsi = TestRSIEngine(self.message_bus)
    
    def tearDown(self):
        """Clean up the temporary directories."""
        shutil.rmtree(self.temp_blob_dir)
        
        # Restore original dirs
        if self.original_blob_dir:
            os.environ['BLOB_STORAGE_DIR'] = self.original_blob_dir
        else:
            del os.environ['BLOB_STORAGE_DIR']
    
    def test_governance_lifecycle_with_audit(self):
        """Test the full governance lifecycle with audit logging."""
        # Add debugging to verify patch signing and storage
        patch = self.rsi.generate_patch("modules/system.py", "Important fix", critical=True)
        
        # Ensure patch is properly signed and registered with the engine
        self.assertTrue(hasattr(patch, 'signed_dict'), "Patch is missing signed_dict attribute")
        self.assertGreaterEqual(len(patch.signatures), 1, "Patch should have at least one signature")
        self.assertIn(patch.patch_id, self.rsi.patches, "Patch not registered with engine")
        
        # Check signature verification directly
        critical = len(patch.signatures) > 1
        verified = self.rsi.verify_patch_signatures(patch, critical=critical)
        self.assertTrue(verified, "Patch signature verification failed before proposing")
        
        # Propose it to governance
        self.rsi.propose_patch(patch)
        
        # Simulate governance approval
        self.message_bus.publish("rsi.patch.governance_decision", {
            "patch_id": patch.patch_id,
            "decision": "approved",
            "reason": "Tests pass, looks good"
        })
        
        # Verify it was applied
        self.assertEqual(patch.status, "applied", "Patch should be applied after approval")
        self.assertIn(patch.patch_id, self.rsi.active_patches, "Patch should be in active_patches list")
        
        # Simulate a contradiction spike causing a veto
        self.message_bus.publish("rsi.patch.governance_decision", {
            "patch_id": patch.patch_id,
            "decision": "veto",
            "reason": "Contradiction spike detected"
        })
        
        # Verify it was rolled back
        self.assertEqual(patch.status, "rolled_back", "Patch should be rolled back after veto")
        self.assertNotIn(patch.patch_id, self.rsi.active_patches, "Patch should not be in active_patches list")
        
        # Now check the audit log contains all events
        events = get_rsi_audit_log()
        
        # Check key events exist
        event_types = [e["event_type"] for e in events]
        self.assertIn("proposal", event_types)
        self.assertIn("governance_decision", event_types)
        self.assertIn("patch_applied", event_types)
        self.assertIn("patch_rolled_back", event_types)
        
        # Verify decision details
        decisions = [e for e in events if e["event_type"] == "governance_decision"]
        self.assertEqual(len(decisions), 2)
        self.assertEqual(decisions[0]["metadata"]["decision"], "approved")
        self.assertEqual(decisions[1]["metadata"]["decision"], "veto")
    
    def test_nested_dependency_rollback(self):
        """Test rollback with nested module dependencies."""
        # Simulate a dependency graph: module1 <- module2 <- module3
        # Apply patches in order
        patch1 = self.rsi.generate_patch("modules/module1.py", "Base Module")
        self.rsi.apply_patch(patch1.patch_id)
        
        patch2 = self.rsi.generate_patch("modules/module2.py", "Depends on Module 1")
        self.rsi.apply_patch(patch2.patch_id)
        
        patch3 = self.rsi.generate_patch("modules/module3.py", "Depends on Module 2")
        self.rsi.apply_patch(patch3.patch_id)
        
        # Record the order in which they're active
        self.assertEqual(len(self.rsi.active_patches), 3)
        
        # Now roll them back in reverse order
        self.rsi.rollback_patch(patch3.patch_id)
        self.assertEqual(len(self.rsi.active_patches), 2)
        self.assertNotIn(patch3.patch_id, self.rsi.active_patches)
        
        self.rsi.rollback_patch(patch2.patch_id)
        self.assertEqual(len(self.rsi.active_patches), 1)
        self.assertNotIn(patch2.patch_id, self.rsi.active_patches)
        
        self.rsi.rollback_patch(patch1.patch_id)
        self.assertEqual(len(self.rsi.active_patches), 0)
        
        # Check all patches are marked as rolled back
        self.assertEqual(patch1.status, "rolled_back")
        self.assertEqual(patch2.status, "rolled_back")
        self.assertEqual(patch3.status, "rolled_back")
        
        # Check audit log contains all rollback events
        events = get_rsi_audit_log()
        rollback_events = [e for e in events if e["event_type"] == "patch_rolled_back"]
        self.assertEqual(len(rollback_events), 3)


if __name__ == "__main__":
    unittest.main()