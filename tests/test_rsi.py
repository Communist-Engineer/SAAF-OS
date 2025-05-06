#!/usr/bin/env python
"""
Unit tests for the Recursive Self-Improvement (RSI) Engine.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import json

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.rsi.engine import RSIEngine, Patch, DummyRSIEngine


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


class TestPatch(unittest.TestCase):
    """Test cases for the Patch class."""
    
    def test_initialization(self):
        """Test patch initialization."""
        # Create a patch
        patch = Patch(
            module_path="modules/test.py",
            content="@@ -1,3 +1,3 @@\n-old line\n+new line\n context",
            description="Test patch",
            author="Test Author"
        )
        
        # Check attributes
        self.assertEqual(patch.module_path, "modules/test.py")
        self.assertEqual(patch.content, "@@ -1,3 +1,3 @@\n-old line\n+new line\n context")
        self.assertEqual(patch.description, "Test patch")
        self.assertEqual(patch.author, "Test Author")
        self.assertEqual(patch.status, "proposed")
        
        # Check UUID was generated
        self.assertTrue(patch.patch_id)
    
    def test_to_dict(self):
        """Test converting patch to dictionary."""
        # Create a patch with a predefined ID
        patch = Patch(
            module_path="modules/test.py",
            content="@@ -1,3 +1,3 @@\n-old line\n+new line\n context",
            description="Test patch",
            author="Test Author",
            patch_id="test-123"
        )
        
        # Convert to dict
        patch_dict = patch.to_dict()
        
        # Check dict values
        self.assertEqual(patch_dict["patch_id"], "test-123")
        self.assertEqual(patch_dict["module_path"], "modules/test.py")
        self.assertEqual(patch_dict["content"], "@@ -1,3 +1,3 @@\n-old line\n+new line\n context")
        self.assertEqual(patch_dict["description"], "Test patch")
        self.assertEqual(patch_dict["author"], "Test Author")
        self.assertEqual(patch_dict["status"], "proposed")
    
    def test_from_dict(self):
        """Test creating patch from dictionary."""
        # Create a patch dictionary
        patch_dict = {
            "patch_id": "test-456",
            "module_path": "modules/another.py",
            "content": "@@ -1,3 +1,3 @@\n-old code\n+new code\n context",
            "description": "Another test patch",
            "author": "Another Author",
            "creation_time": 1620000000.0,
            "status": "approved"
        }
        
        # Create patch from dict
        patch = Patch.from_dict(patch_dict)
        
        # Check attributes
        self.assertEqual(patch.patch_id, "test-456")
        self.assertEqual(patch.module_path, "modules/another.py")
        self.assertEqual(patch.content, "@@ -1,3 +1,3 @@\n-old code\n+new code\n context")
        self.assertEqual(patch.description, "Another test patch")
        self.assertEqual(patch.author, "Another Author")
        self.assertEqual(patch.creation_time, 1620000000.0)
        self.assertEqual(patch.status, "approved")


class TestRSIEngine(unittest.TestCase):
    """Test cases for the RSIEngine class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.message_bus = MockMessageBus()
        self.rsi = RSIEngine(self.message_bus)
    
    def test_initialization(self):
        """Test RSI engine initialization."""
        self.assertEqual(self.rsi.message_bus, self.message_bus)
        self.assertEqual(self.rsi.patches, {})
        self.assertEqual(self.rsi.rollback_table, {})
        self.assertEqual(self.rsi.active_patches, [])
    
    def test_generate_patch(self):
        """Test patch generation."""
        # Generate a patch
        module_path = "modules/test_module.py"
        description = "Test description"
        patch = self.rsi.generate_patch(module_path, description)
        
        # Check patch was generated and stored
        self.assertEqual(patch.module_path, module_path)
        self.assertEqual(patch.description, description)
        self.assertEqual(self.rsi.patches[patch.patch_id], patch)
    
    def test_propose_patch(self):
        """Test patch proposal."""
        # Generate a patch
        patch = self.rsi.generate_patch("modules/test.py", "Test patch")
        
        # Propose the patch
        result = self.rsi.propose_patch(patch)
        
        # Check results
        self.assertTrue(result)
        
        # Check a message was published
        published = self.message_bus.get_published_messages("rsi.patch.proposed")
        self.assertEqual(len(published), 1)
        self.assertEqual(published[0]["patch"]["patch_id"], patch.patch_id)
    
    def test_apply_patch(self):
        """Test patch application."""
        # Generate a patch
        patch = self.rsi.generate_patch("modules/test.py", "Test patch")
        
        # Apply the patch
        result = self.rsi.apply_patch(patch.patch_id)
        
        # Check results
        self.assertTrue(result)
        self.assertEqual(patch.status, "applied")
        self.assertIn(patch.patch_id, self.rsi.active_patches)
        self.assertIn(patch.patch_id, self.rsi.rollback_table)
        
        # Check a message was published
        published = self.message_bus.get_published_messages("rsi.patch.applied")
        self.assertEqual(len(published), 1)
        self.assertEqual(published[0]["patch_id"], patch.patch_id)
        self.assertTrue(published[0]["success"])
    
    def test_rollback_patch(self):
        """Test patch rollback."""
        # Generate and apply a patch
        patch = self.rsi.generate_patch("modules/test.py", "Test patch")
        self.rsi.apply_patch(patch.patch_id)
        
        # Now roll it back
        result = self.rsi.rollback_patch(patch.patch_id)
        
        # Check results
        self.assertTrue(result)
        self.assertEqual(patch.status, "rolled_back")
        self.assertNotIn(patch.patch_id, self.rsi.active_patches)
        
        # Check a message was published
        published = self.message_bus.get_published_messages("rsi.patch.rolled_back")
        self.assertEqual(len(published), 1)
        self.assertEqual(published[0]["patch_id"], patch.patch_id)
        self.assertTrue(published[0]["success"])
    
    def test_handle_governance_approval(self):
        """Test handling governance approval."""
        # Generate a patch
        patch = self.rsi.generate_patch("modules/test.py", "Test patch")
        
        # Send a governance approval message
        self.message_bus.publish("rsi.patch.governance_decision", {
            "patch_id": patch.patch_id,
            "decision": "approved"
        })
        
        # Check the patch was applied
        self.assertEqual(patch.status, "approved")
        self.assertIn(patch.patch_id, self.rsi.active_patches)
    
    def test_handle_governance_rejection(self):
        """Test handling governance rejection."""
        # Generate a patch
        patch = self.rsi.generate_patch("modules/test.py", "Test patch")
        
        # Send a governance rejection message
        self.message_bus.publish("rsi.patch.governance_decision", {
            "patch_id": patch.patch_id,
            "decision": "rejected"
        })
        
        # Check the patch was rejected
        self.assertEqual(patch.status, "rejected")
        self.assertNotIn(patch.patch_id, self.rsi.active_patches)
    
    def test_handle_governance_veto(self):
        """Test handling governance veto for an applied patch."""
        # Generate and apply a patch
        patch = self.rsi.generate_patch("modules/test.py", "Test patch")
        self.rsi.apply_patch(patch.patch_id)
        
        # Send a governance veto message
        self.message_bus.publish("rsi.patch.governance_decision", {
            "patch_id": patch.patch_id,
            "decision": "veto"
        })
        
        # Check the patch was rolled back
        self.assertEqual(patch.status, "rolled_back")
        self.assertNotIn(patch.patch_id, self.rsi.active_patches)
        
        # Check rollback message was published
        published = self.message_bus.get_published_messages("rsi.patch.rolled_back")
        self.assertEqual(len(published), 1)
    
    def test_veto_and_rollback_scenario(self):
        """Test the complete veto and rollback scenario."""
        # Generate and apply a patch
        patch = self.rsi.generate_patch("modules/core.py", "Critical enhancement")
        
        # Propose the patch
        self.rsi.propose_patch(patch)
        
        # Approve the patch through governance
        self.message_bus.publish("rsi.patch.governance_decision", {
            "patch_id": patch.patch_id,
            "decision": "approved"
        })
        
        # Verify it was applied
        self.assertEqual(patch.status, "applied")
        self.assertIn(patch.patch_id, self.rsi.active_patches)
        
        # Now veto the patch
        self.message_bus.publish("rsi.patch.governance_decision", {
            "patch_id": patch.patch_id,
            "decision": "veto"
        })
        
        # Verify it was rolled back
        self.assertEqual(patch.status, "rolled_back")
        self.assertNotIn(patch.patch_id, self.rsi.active_patches)
        
        # Check rollback message was published
        published = self.message_bus.get_published_messages("rsi.patch.rolled_back")
        self.assertTrue(any(msg.get("success", False) for msg in published))


class TestDummyRSIEngine(unittest.TestCase):
    """Test cases for the DummyRSIEngine class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.message_bus = MockMessageBus()
        self.rsi = DummyRSIEngine(self.message_bus)
    
    def test_generate_deterministic_patch(self):
        """Test dummy patch generation."""
        # Generate patches for the same module path
        patch1 = self.rsi.generate_patch("modules/test.py")
        patch2 = self.rsi.generate_patch("modules/test.py")
        
        # Content should be deterministic but IDs should be different
        self.assertEqual(patch1.content, patch2.content)
        self.assertNotEqual(patch1.patch_id, patch2.patch_id)
        self.assertTrue(patch1.patch_id.startswith("test_patch_"))
        self.assertTrue(patch2.patch_id.startswith("test_patch_"))


import os
import pytest
from modules.rsi.engine import RSIEngine, Patch
from modules.rsi.crypto_utils import sign_patch, verify_patch_signature

def test_patch_signature_valid():
    engine = RSIEngine()
    patch = engine.generate_patch("modules/test_module.py", description="Test patch")
    # Should verify with correct signature
    assert engine.verify_patch_signatures(patch) is True

def test_patch_signature_tampered():
    engine = RSIEngine()
    patch = engine.generate_patch("modules/test_module.py", description="Test patch")
    # Tamper with patch content
    patch.content = "@@ -1,3 +1,3 @@\n-old line\n+TAMPERED\n context"
    # Should fail verification
    assert engine.verify_patch_signatures(patch) is False

def test_critical_patch_requires_two_signatures(monkeypatch):
    engine = RSIEngine()
    # Simulate presence of second key
    monkeypatch.setattr(os.path, "exists", lambda path: True)
    patch = engine.generate_patch("modules/test_module.py", description="Critical patch", critical=True)
    # Should have two signatures
    assert len(patch.signatures) == 2
    # Should verify with both signatures
    assert engine.verify_patch_signatures(patch, critical=True) is True

def test_critical_patch_missing_second_signature():
    engine = RSIEngine()
    patch = engine.generate_patch("modules/test_module.py", description="Critical patch", critical=True)
    # Remove second signature
    patch.signatures = patch.signatures[:1]
    # Should fail verification for critical
    assert engine.verify_patch_signatures(patch, critical=True) is False


if __name__ == "__main__":
    unittest.main()