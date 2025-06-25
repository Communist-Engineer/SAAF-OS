"""
Message Bus Adapter for SAAF-OS

This module implements the Message Bus adapter as specified in message_bus_spec.md.
It provides a communication substrate for all SAAF-OS components using 
a publish/subscribe pattern over ZeroMQ.
"""

import uuid
import json
import time
import threading
import queue
import logging
from typing import Dict, List, Any, Optional, Callable, Union

try:
    import zmq
    ZEROMQ_AVAILABLE = True
except ImportError:
    ZEROMQ_AVAILABLE = False
    logging.warning("ZeroMQ not available. Using in-memory message bus instead.")

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MessageBus")


class MessageBusAdapter:
    """
    Adapter class for the Message Bus component, implementing the 
    publish/subscribe pattern for inter-module communication.
    """
    
    def __init__(self, module_name: str, address: str = "tcp://127.0.0.1", port: int = 5555):
        """
        Initialize a Message Bus adapter.
        
        Args:
            module_name: Name of the module using this adapter
            address: ZeroMQ address to connect to
            port: ZeroMQ port to use
        """
        self.module_name = module_name
        self.address = address
        self.port = port
        
        # Message history for debugging and tests
        self.message_history: List[Dict[str, Any]] = []
        
        # Callbacks for subscribed topics
        self.callbacks: Dict[str, List[Callable]] = {}
        
        # Set up ZeroMQ if available, otherwise use in-memory bus
        if ZEROMQ_AVAILABLE:
            self._setup_zeromq()
        else:
            self._setup_in_memory()
        
        logger.info(f"Initialized MessageBusAdapter for module: {module_name}")
    
    def _setup_zeromq(self):
        """Set up ZeroMQ publisher and subscriber."""
        # Create context
        self.context = zmq.Context()
        
        # Set up publisher socket
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.bind(f"{self.address}:{self.port}")
        
        # Set up subscriber socket
        self.subscriber = self.context.socket(zmq.SUB)
        self.subscriber.connect(f"{self.address}:{self.port}")
        
        # Start subscriber thread
        self.running = True
        self.subscriber_thread = threading.Thread(target=self._subscriber_loop)
        self.subscriber_thread.daemon = True
        self.subscriber_thread.start()
    
    def _setup_in_memory(self):
        """Set up in-memory message bus."""
        # Use the global in-memory bus provided by MessageBusFactory
        self.publisher = None
        self.subscriber = None
        self.running = True
        
        # Register with the factory
        MessageBusFactory._register_adapter(self)
    
    def _subscriber_loop(self):
        """Background thread for receiving messages."""
        while self.running:
            try:
                # Check if there are any messages
                if ZEROMQ_AVAILABLE:
                    try:
                        # Use non-blocking receive with timeout
                        topic, msg_data = self.subscriber.recv_multipart(zmq.NOBLOCK)
                        msg = json.loads(msg_data.decode())
                        
                        # Skip messages from self
                        if msg["header"]["sender"] != self.module_name:
                            self._handle_message(topic.decode(), msg)
                    except zmq.Again:
                        # No messages available
                        time.sleep(0.01)
            except Exception as e:
                logger.error(f"Error in subscriber loop: {e}")
                time.sleep(0.1)
    
    def _handle_message(self, topic: str, message: Dict):
        """
        Handle incoming messages from the bus.
        
        Args:
            topic: Message topic
            message: Message data
        """
        if topic in self.callbacks:
            for callback in self.callbacks[topic]:
                try:
                    callback(message)
                except Exception as e:
                    logger.error(f"Error in callback for topic {topic}: {e}")
    
    def publish(self, topic: str, data: Dict[str, Any]) -> str:
        """
        Publish a message to the bus.
        
        Args:
            topic: Message topic
            data: Message data
            
        Returns:
            Message ID
        """
        # Generate message ID
        msg_id = str(uuid.uuid4())
        
        # Create message structure
        message = {
            "header": {
                "id": msg_id,
                "timestamp": time.time(),
                "sender": self.module_name,
                "topic": topic,
                "version": "1.0"
            },
            "payload": {
                "data": data
            }
        }
        
        # Store in history
        self.message_history.append(message)
        
        # Publish to bus
        if ZEROMQ_AVAILABLE and self.publisher:
            self.publisher.send_multipart([
                topic.encode(),
                json.dumps(message).encode()
            ])
        else:
            # Use in-memory bus
            MessageBusFactory._broadcast_message(self, topic, message)
        
        return msg_id
    
    def subscribe(self, topic: str, callback: Callable[[Dict], None]) -> None:
        """
        Subscribe to a topic on the bus.
        
        Args:
            topic: Topic to subscribe to
            callback: Function to call when a message is received
        """
        # Register callback
        if topic not in self.callbacks:
            self.callbacks[topic] = []
        self.callbacks[topic].append(callback)
        
        # Subscribe in ZeroMQ if available
        if ZEROMQ_AVAILABLE and self.subscriber:
            self.subscriber.setsockopt(zmq.SUBSCRIBE, topic.encode())
        
        logger.debug(f"Module {self.module_name} subscribed to topic: {topic}")
    
    def unsubscribe(self, topic: str, callback: Optional[Callable] = None) -> None:
        """
        Unsubscribe from a topic.
        
        Args:
            topic: Topic to unsubscribe from
            callback: Specific callback to remove, or None to remove all
        """
        if topic in self.callbacks:
            if callback is None:
                # Remove all callbacks
                self.callbacks.pop(topic)
                
                # Unsubscribe in ZeroMQ if available
                if ZEROMQ_AVAILABLE and self.subscriber:
                    self.subscriber.setsockopt(zmq.UNSUBSCRIBE, topic.encode())
            else:
                # Remove specific callback
                if callback in self.callbacks[topic]:
                    self.callbacks[topic].remove(callback)
                
                # If no callbacks left, unsubscribe completely
                if not self.callbacks[topic]:
                    self.callbacks.pop(topic)
                    
                    # Unsubscribe in ZeroMQ if available
                    if ZEROMQ_AVAILABLE and self.subscriber:
                        self.subscriber.setsockopt(zmq.UNSUBSCRIBE, topic.encode())
    
    def get_message_history(self, topic: Optional[str] = None, limit: Optional[int] = None) -> List[Dict]:
        """
        Get message history, optionally filtered by topic.
        
        Args:
            topic: Topic to filter by, or None for all topics
            limit: Maximum number of messages to return, or None for all
            
        Returns:
            List of messages
        """
        messages = self.message_history
        
        # Filter by topic if specified
        if topic:
            messages = [m for m in messages if m["header"]["topic"] == topic]
        
        # Apply limit
        if limit and limit > 0:
            messages = messages[-limit:]
        
        return messages
    
    def shutdown(self) -> None:
        """Shut down the adapter."""
        self.running = False
        
        # Clean up ZeroMQ resources if available
        if ZEROMQ_AVAILABLE:
            if hasattr(self, 'subscriber') and self.subscriber:
                self.subscriber.close()
            if hasattr(self, 'publisher') and self.publisher:
                self.publisher.close()
            if hasattr(self, 'context') and self.context:
                self.context.term()
        
        # Unregister from factory
        MessageBusFactory._unregister_adapter(self)
        
        logger.info(f"Shut down MessageBusAdapter for module: {self.module_name}")


class MessageBusFactory:
    """
    Factory class for MessageBusAdapters with in-memory message bus capabilities.
    This allows modules to communicate even when ZeroMQ is not available.
    """
    
    # Dictionary of module name to adapter instance
    _instances: Dict[str, MessageBusAdapter] = {}
    _next_port = 5555
    
    @classmethod
    def get_adapter(cls, module_name: str) -> MessageBusAdapter:
        """
        Get a MessageBusAdapter for a module, creating it if it doesn't exist.
        
        Args:
            module_name: Name of the module
            
        Returns:
            MessageBusAdapter instance
        """
        if module_name not in cls._instances:
            adapter = MessageBusAdapter(module_name, port=cls._next_port)
            cls._instances[module_name] = adapter
            cls._next_port += 1
        
        return cls._instances[module_name]
    
    @classmethod
    def _register_adapter(cls, adapter: MessageBusAdapter) -> None:
        """
        Register an adapter with the factory.
        
        Args:
            adapter: Adapter to register
        """
        cls._instances[adapter.module_name] = adapter
    
    @classmethod
    def _unregister_adapter(cls, adapter: MessageBusAdapter) -> None:
        """
        Unregister an adapter from the factory.
        
        Args:
            adapter: Adapter to unregister
        """
        if adapter.module_name in cls._instances:
            cls._instances.pop(adapter.module_name)
    
    @classmethod
    def _broadcast_message(cls, sender: MessageBusAdapter, topic: str, message: Dict) -> None:
        """
        Broadcast a message to all adapters except the sender.
        Used for in-memory message bus.
        
        Args:
            sender: Adapter that sent the message
            topic: Message topic
            message: Message data
        """
        for module_name, adapter in cls._instances.items():
            if adapter != sender:
                adapter._handle_message(topic, message)
    
    @classmethod
    def shutdown_all(cls) -> None:
        """Shut down all adapters."""
        # Make a copy of the keys to avoid modifying during iteration
        module_names = list(cls._instances.keys())
        for module_name in module_names:
            if module_name in cls._instances:
                cls._instances[module_name].shutdown()
        
        # Clear instances
        cls._instances.clear()