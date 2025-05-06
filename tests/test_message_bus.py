#!/usr/bin/env python
"""
Unit tests for the Message Bus adapter (bus/adapter.py)
"""

import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch
import threading

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bus.adapter import MessageBusAdapter, MessageBusFactory


class TestMessageBusAdapter(unittest.TestCase):
    """Test cases for the MessageBusAdapter class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create an adapter for testing
        self.adapter = MessageBusAdapter("test_module")
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.adapter.shutdown()
    
    def test_initialization(self):
        """Test adapter initialization."""
        self.assertEqual(self.adapter.module_name, "test_module")
        self.assertEqual(self.adapter.address, "tcp://127.0.0.1")
        self.assertTrue(hasattr(self.adapter, 'publisher'))
        self.assertTrue(hasattr(self.adapter, 'subscriber'))
        self.assertTrue(hasattr(self.adapter, 'callbacks'))
        self.assertTrue(hasattr(self.adapter, 'message_history'))
    
    def test_publish(self):
        """Test publishing a message."""
        msg_id = self.adapter.publish("test.topic", {"key": "value"})
        self.assertTrue(len(msg_id) > 0)  # UUID should be non-empty
        self.assertEqual(len(self.adapter.message_history), 1)
        
        # Check message structure
        msg = self.adapter.message_history[0]
        self.assertEqual(msg["header"]["sender"], "test_module")
        self.assertEqual(msg["header"]["topic"], "test.topic")
        self.assertEqual(msg["payload"]["data"], {"key": "value"})
    
    def test_get_message_history(self):
        """Test retrieving message history."""
        # Publish some messages
        self.adapter.publish("topic1", {"data": 1})
        self.adapter.publish("topic2", {"data": 2})
        self.adapter.publish("topic1", {"data": 3})
        
        # Test retrieving all messages
        all_msgs = self.adapter.get_message_history()
        self.assertEqual(len(all_msgs), 3)
        
        # Test filtering by topic
        topic1_msgs = self.adapter.get_message_history(topic="topic1")
        self.assertEqual(len(topic1_msgs), 2)
        self.assertEqual(topic1_msgs[0]["payload"]["data"]["data"], 1)
        self.assertEqual(topic1_msgs[1]["payload"]["data"]["data"], 3)
        
        # Test limit
        limited_msgs = self.adapter.get_message_history(limit=2)
        self.assertEqual(len(limited_msgs), 2)
        self.assertEqual(limited_msgs[1]["payload"]["data"]["data"], 3)
    
    @patch('zmq.Context')
    def test_subscribe_and_callback(self, mock_context):
        """Test subscribing to a topic and receiving callbacks."""
        # Mock the ZeroMQ components for testing
        mock_sub = MagicMock()
        mock_context.return_value.socket.return_value = mock_sub
        
        # Create a new adapter with mocked ZeroMQ
        adapter = MessageBusAdapter("test_module")
        
        # Create a callback function
        callback = MagicMock()
        
        # Subscribe to a topic
        adapter.subscribe("test.topic", callback)
        
        # Verify that we set the subscription
        mock_sub.setsockopt.assert_called_with(
            MagicMock(), b"test.topic"
        )
        
        # Check that the callback was registered
        self.assertEqual(len(adapter.callbacks), 1)
        self.assertEqual(len(adapter.callbacks["test.topic"]), 1)
        self.assertEqual(adapter.callbacks["test.topic"][0], callback)
        
        adapter.shutdown()
    
    def test_unsubscribe(self):
        """Test unsubscribing from a topic."""
        # Subscribe to a topic
        callback = MagicMock()
        self.adapter.subscribe("test.topic", callback)
        
        # Verify subscription was registered
        self.assertEqual(len(self.adapter.callbacks["test.topic"]), 1)
        
        # Unsubscribe
        self.adapter.unsubscribe("test.topic", callback)
        
        # Verify callback was removed
        self.assertEqual(len(self.adapter.callbacks["test.topic"]), 0)
        
        # Subscribe again
        self.adapter.subscribe("test.topic", callback)
        
        # Unsubscribe without specifying callback (should remove all)
        self.adapter.unsubscribe("test.topic")
        
        # Verify topic was removed
        self.assertNotIn("test.topic", self.adapter.callbacks)


class TestMessageBusFactory(unittest.TestCase):
    """Test cases for the MessageBusFactory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Clear any existing instances
        MessageBusFactory._instances = {}
    
    def tearDown(self):
        """Tear down test fixtures."""
        MessageBusFactory.shutdown_all()
    
    def test_get_adapter(self):
        """Test getting adapters from the factory."""
        # Get an adapter
        adapter1 = MessageBusFactory.get_adapter("module1")
        self.assertEqual(adapter1.module_name, "module1")
        
        # Get the same adapter again (should return the same instance)
        adapter1_again = MessageBusFactory.get_adapter("module1")
        self.assertIs(adapter1, adapter1_again)
        
        # Get a different adapter
        adapter2 = MessageBusFactory.get_adapter("module2")
        self.assertEqual(adapter2.module_name, "module2")
        self.assertIsNot(adapter1, adapter2)
    
    def test_shutdown_all(self):
        """Test shutting down all adapters."""
        # Create some adapters
        adapter1 = MessageBusFactory.get_adapter("module1")
        adapter2 = MessageBusFactory.get_adapter("module2")
        
        # Check that they're in the instances dictionary
        self.assertEqual(len(MessageBusFactory._instances), 2)
        
        # Shut down all adapters
        MessageBusFactory.shutdown_all()
        
        # Check that the instances dictionary is empty
        self.assertEqual(len(MessageBusFactory._instances), 0)


if __name__ == '__main__':
    unittest.main()