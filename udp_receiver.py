"""
UDP Data Receiver Module for Real-Time Aircraft Classification
"""

import socket
import struct
import pandas as pd
import numpy as np
from datetime import datetime
import threading
import queue
import time


class DataSource:
    """Base class for data sources"""
    def __init__(self):
        self.is_running = False
        self.data_queue = queue.Queue()
        
    def start(self):
        raise NotImplementedError
        
    def stop(self):
        raise NotImplementedError
        
    def get_received_data(self):
        data_list = []
        while not self.data_queue.empty():
            try:
                data_list.append(self.data_queue.get_nowait())
            except queue.Empty:
                break
        return data_list
    
    def has_data(self):
        return not self.data_queue.empty()


class CSVDataSender(DataSource):
    """Send CSV data row-by-row via UDP"""
    
    def __init__(self, csv_path, target_host='127.0.0.1', target_port=5005, interval=0.1):
        super().__init__()
        self.csv_path = csv_path
        self.target_host = target_host
        self.target_port = target_port
        self.interval = interval
        self.sender_thread = None
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
    def start(self):
        if self.is_running:
            return False
            
        self.is_running = True
        self.sender_thread = threading.Thread(target=self._send_loop, daemon=True)
        self.sender_thread.start()
        return True
    
    def stop(self):
        self.is_running = False
        if self.sender_thread:
            self.sender_thread.join(timeout=2)
        try:
            self.socket.close()
        except:
            pass
    
    def _send_loop(self):
        try:
            df = pd.read_csv(self.csv_path)
            required_cols = ['Time', 'Height', 'Resultant_acceleration', 'Resultant_velocity', 'AGC']
            
            if not all(col in df.columns for col in required_cols):
                print(f"Error: CSV missing required columns: {required_cols}")
                self.is_running = False
                return
            
            for _, row in df.iterrows():
                if not self.is_running:
                    break
                
                packet = struct.pack(
                    '<ffffi',
                    float(row['Time']),
                    float(row['Height']),
                    float(row['Resultant_acceleration']),
                    float(row['Resultant_velocity']),
                    int(row['AGC'])
                )
                
                self.socket.sendto(packet, (self.target_host, self.target_port))
                time.sleep(self.interval)
            
            self.is_running = False
            
        except Exception as e:
            print(f"Error sending CSV data: {e}")
            self.is_running = False


class UDPDataReceiver(DataSource):
    """Real-time UDP data receiver"""
    
    def __init__(self, host='0.0.0.0', port=5005, buffer_size=4096):
        super().__init__()
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.socket = None
        self.receiver_thread = None
        self.last_received_time = None
        
    def start(self):
        if self.is_running:
            return False
        
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            # ✅ IMPORTANT FIX (prevents WinError 10048)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            self.socket.bind((self.host, self.port))
            self.socket.settimeout(1.0)

            self.is_running = True
            
            self.receiver_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.receiver_thread.start()
            
            return True

        except Exception as e:
            print(f"Error starting UDP receiver: {e}")
            return False
    
    def stop(self):
        self.is_running = False

        if self.receiver_thread:
            self.receiver_thread.join(timeout=2)

        if self.socket:
            try:
                self.socket.close()
            except:
                pass

        self.socket = None
    
    def _receive_loop(self):
        while self.is_running:
            try:
                data, addr = self.socket.recvfrom(self.buffer_size)

                if data:
                    decoded_data = self._decode_dat_packet(data)

                    if decoded_data:
                        self.last_received_time = datetime.now()
                        self.data_queue.put({
                            'timestamp': self.last_received_time,
                            'source': addr,
                            'data': decoded_data
                        })

            except socket.timeout:
                continue

            except Exception as e:
                if self.is_running:
                    print(f"Error receiving data: {e}")
    
    def _decode_dat_packet(self, raw_data):
        try:
            records = []
            record_size = 20
            
            for i in range(0, len(raw_data), record_size):
                if i + record_size <= len(raw_data):
                    chunk = raw_data[i:i+record_size]
                    unpacked = struct.unpack('<ffffi', chunk)
                    
                    record = {
                        'Time': float(unpacked[0]),
                        'Height': float(unpacked[1]),
                        'Resultant_acceleration': float(unpacked[2]),
                        'Resultant_velocity': float(unpacked[3]),
                        'AGC': int(unpacked[4])
                    }
                    records.append(record)
            
            return records if records else None
            
        except Exception as e:
            print(f"Error decoding packet: {e}")
            return None
    
    def get_last_received_time(self):
        return self.last_received_time


class SlidingWindowBuffer:
    """Manages sliding window buffer for real-time data processing"""
    
    def __init__(self, window_size=10, step_size=1):
        self.window_size = window_size
        self.step_size = step_size
        self.all_data = []
        self.current_window = []
        self.row_counter = 0
        
    def set_window_size(self, size):
        self.window_size = size
    
    def set_step_size(self, size):
        self.step_size = size
    
    def add_record(self, record):
        self.row_counter += 1
        record['Row_No'] = self.row_counter
        self.all_data.append(record)
        
        self.current_window.append(record)

        if len(self.current_window) > self.window_size:
            for _ in range(self.step_size):
                if len(self.current_window) > self.window_size:
                    self.current_window.pop(0)
    
    def get_current_window(self):
        return pd.DataFrame(self.current_window) if self.current_window else None
    
    def get_all_data(self):
        return pd.DataFrame(self.all_data) if self.all_data else None
    
    def is_window_ready(self):
        return len(self.current_window) >= self.window_size
    
    def clear(self):
        self.all_data = []
        self.current_window = []
        self.row_counter = 0