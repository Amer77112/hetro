"""
WebSocket Handler for Real-time Communication
Manages real-time updates between the FL system and dashboard.
"""

import json
import logging
import threading
import time
from typing import Dict, Any, Callable, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class WebSocketHandler:
    """Handles WebSocket communication for real-time FL updates."""
    
    def __init__(self, socketio_instance=None):
        self.socketio = socketio_instance
        self.connected_clients = set()
        self.update_queue = []
        self.update_lock = threading.Lock()
        self.is_broadcasting = False
        
    def set_socketio(self, socketio_instance):
        """Set the SocketIO instance."""
        self.socketio = socketio_instance
        
    def emit_status_update(self, status_data: Dict[str, Any]):
        """Emit status update to all connected clients."""
        if self.socketio:
            self.socketio.emit('status_update', status_data)
            logger.debug(f"Emitted status update: {status_data.get('fl_status', 'unknown')}")
    
    def emit_round_start(self, round_num: int):
        """Emit round start notification."""
        if self.socketio:
            self.socketio.emit('round_start', {'round': round_num, 'timestamp': datetime.now().isoformat()})
            logger.info(f"Emitted round start for round {round_num}")
    
    def emit_round_complete(self, round_num: int, success: bool, metrics: Dict = None):
        """Emit round completion notification."""
        if self.socketio:
            data = {
                'round': round_num,
                'success': success,
                'timestamp': datetime.now().isoformat()
            }
            if metrics:
                data['metrics'] = metrics
            
            self.socketio.emit('round_complete', data)
            logger.info(f"Emitted round complete for round {round_num} (success: {success})")
    
    def emit_metrics_update(self, metrics_data: Dict[str, Any]):
        """Emit metrics update to dashboard."""
        if self.socketio:
            self.socketio.emit('metrics_update', metrics_data)
            logger.debug("Emitted metrics update")
    
    def emit_model_update(self, model_name: str, model_metrics: Dict[str, Any]):
        """Emit individual model update."""
        if self.socketio:
            data = {
                'model_name': model_name,
                'metrics': model_metrics,
                'timestamp': datetime.now().isoformat()
            }
            self.socketio.emit('model_update', data)
            logger.debug(f"Emitted model update for {model_name}")
    
    def emit_aggregation_update(self, aggregation_stats: Dict[str, Any]):
        """Emit aggregation process update."""
        if self.socketio:
            data = {
                'aggregation_stats': aggregation_stats,
                'timestamp': datetime.now().isoformat()
            }
            self.socketio.emit('aggregation_update', data)
            logger.debug("Emitted aggregation update")
    
    def emit_error(self, error_message: str, error_type: str = 'general'):
        """Emit error notification."""
        if self.socketio:
            data = {
                'error': error_message,
                'error_type': error_type,
                'timestamp': datetime.now().isoformat()
            }
            self.socketio.emit('fl_error', data)
            logger.error(f"Emitted error: {error_message}")
    
    def emit_log_message(self, message: str, level: str = 'info'):
        """Emit log message to dashboard."""
        if self.socketio:
            data = {
                'message': message,
                'level': level,
                'timestamp': datetime.now().isoformat()
            }
            self.socketio.emit('log_message', data)
    
    def emit_fl_complete(self, total_rounds: int, final_metrics: Dict = None):
        """Emit FL process completion."""
        if self.socketio:
            data = {
                'total_rounds': total_rounds,
                'timestamp': datetime.now().isoformat()
            }
            if final_metrics:
                data['final_metrics'] = final_metrics
            
            self.socketio.emit('fl_complete', data)
            logger.info(f"Emitted FL completion for {total_rounds} rounds")
    
    def emit_progress_update(self, current_round: int, total_rounds: int, 
                           round_progress: float = None):
        """Emit progress update."""
        if self.socketio:
            progress = (current_round / total_rounds) * 100 if total_rounds > 0 else 0
            
            data = {
                'current_round': current_round,
                'total_rounds': total_rounds,
                'overall_progress': progress,
                'round_progress': round_progress,
                'timestamp': datetime.now().isoformat()
            }
            
            self.socketio.emit('progress_update', data)
    
    def queue_update(self, update_type: str, data: Dict[str, Any]):
        """Queue an update for batch processing."""
        with self.update_lock:
            self.update_queue.append({
                'type': update_type,
                'data': data,
                'timestamp': datetime.now().isoformat()
            })
    
    def start_batch_updates(self, interval: float = 1.0):
        """Start batch update processing in a separate thread."""
        if self.is_broadcasting:
            return
        
        self.is_broadcasting = True
        broadcast_thread = threading.Thread(target=self._batch_update_worker, args=(interval,))
        broadcast_thread.daemon = True
        broadcast_thread.start()
        logger.info("Started batch update broadcasting")
    
    def stop_batch_updates(self):
        """Stop batch update processing."""
        self.is_broadcasting = False
        logger.info("Stopped batch update broadcasting")
    
    def _batch_update_worker(self, interval: float):
        """Worker thread for batch update processing."""
        while self.is_broadcasting:
            try:
                updates_to_send = []
                
                with self.update_lock:
                    if self.update_queue:
                        updates_to_send = self.update_queue.copy()
                        self.update_queue.clear()
                
                if updates_to_send and self.socketio:
                    # Group updates by type and send the most recent ones
                    grouped_updates = {}
                    for update in updates_to_send:
                        update_type = update['type']
                        grouped_updates[update_type] = update
                    
                    # Send grouped updates
                    for update_type, update in grouped_updates.items():
                        self.socketio.emit(update_type, update['data'])
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in batch update worker: {e}")
                time.sleep(interval)
    
    def register_client(self, client_id: str):
        """Register a new client connection."""
        self.connected_clients.add(client_id)
        logger.info(f"Client {client_id} connected. Total clients: {len(self.connected_clients)}")
    
    def unregister_client(self, client_id: str):
        """Unregister a client connection."""
        self.connected_clients.discard(client_id)
        logger.info(f"Client {client_id} disconnected. Total clients: {len(self.connected_clients)}")
    
    def get_connected_clients_count(self) -> int:
        """Get the number of connected clients."""
        return len(self.connected_clients)
    
    def broadcast_system_info(self, system_info: Dict[str, Any]):
        """Broadcast system information to all clients."""
        if self.socketio:
            self.socketio.emit('system_info', system_info)
            logger.debug("Broadcasted system info")


class FLEventEmitter:
    """Event emitter specifically for FL system events."""
    
    def __init__(self, websocket_handler: WebSocketHandler):
        self.ws_handler = websocket_handler
        
    def on_fl_start(self, config: Dict[str, Any]):
        """Handle FL process start."""
        self.ws_handler.emit_status_update({
            'fl_status': 'starting',
            'config': config,
            'timestamp': datetime.now().isoformat()
        })
        self.ws_handler.emit_log_message("Federated learning process started", "info")
    
    def on_models_loaded(self, model_names: list):
        """Handle models loaded event."""
        self.ws_handler.emit_status_update({
            'fl_status': 'models_loaded',
            'models_loaded': model_names,
            'timestamp': datetime.now().isoformat()
        })
        self.ws_handler.emit_log_message(f"Loaded {len(model_names)} models: {', '.join(model_names)}", "info")
    
    def on_round_start(self, round_num: int, total_rounds: int):
        """Handle round start event."""
        self.ws_handler.emit_round_start(round_num)
        self.ws_handler.emit_progress_update(round_num, total_rounds)
        self.ws_handler.emit_log_message(f"Starting round {round_num}/{total_rounds}", "info")
    
    def on_local_prediction(self, model_name: str, metrics: Dict[str, Any]):
        """Handle local model prediction event."""
        self.ws_handler.emit_model_update(model_name, metrics)
        self.ws_handler.emit_log_message(f"Got predictions from {model_name}", "info")
    
    def on_aggregation_complete(self, aggregation_stats: Dict[str, Any]):
        """Handle aggregation completion event."""
        self.ws_handler.emit_aggregation_update(aggregation_stats)
        self.ws_handler.emit_log_message("Aggregation completed", "info")
    
    def on_global_model_update(self, metrics: Dict[str, Any]):
        """Handle global model update event."""
        self.ws_handler.emit_metrics_update({'global_model': metrics})
        self.ws_handler.emit_log_message("Global model updated", "info")
    
    def on_local_model_update(self, model_name: str, update_stats: Dict[str, Any]):
        """Handle local model update event."""
        self.ws_handler.emit_log_message(f"Updated {model_name} with global knowledge", "info")
    
    def on_round_complete(self, round_num: int, success: bool, metrics: Dict = None):
        """Handle round completion event."""
        self.ws_handler.emit_round_complete(round_num, success, metrics)
        status = "successfully" if success else "with errors"
        self.ws_handler.emit_log_message(f"Round {round_num} completed {status}", "info" if success else "warning")
    
    def on_fl_complete(self, total_rounds: int, final_metrics: Dict = None):
        """Handle FL process completion event."""
        self.ws_handler.emit_fl_complete(total_rounds, final_metrics)
        self.ws_handler.emit_status_update({
            'fl_status': 'completed',
            'timestamp': datetime.now().isoformat()
        })
        self.ws_handler.emit_log_message(f"Federated learning completed after {total_rounds} rounds", "info")
    
    def on_error(self, error_message: str, error_type: str = 'general'):
        """Handle error event."""
        self.ws_handler.emit_error(error_message, error_type)
        self.ws_handler.emit_status_update({
            'fl_status': 'error',
            'error': error_message,
            'timestamp': datetime.now().isoformat()
        })


# Global instance for easy access
websocket_handler = WebSocketHandler()
fl_event_emitter = FLEventEmitter(websocket_handler)