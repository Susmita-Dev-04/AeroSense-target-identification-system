"""
udp_dat_processor.py

Actual UDP simulation/process file for airborne target data.

Purpose:
1. Receive UDP packets in .dat binary format.
2. Decode packet format: <ffffi
   Time, Height, Resultant_acceleration, Resultant_velocity, AGC
3. Convert received data into CSV format.
4. Save the structured CSV file for next ML prediction process.

Run:
    python udp_dat_processor.py

Default:
    Host: 0.0.0.0
    Port: 5005
    Output CSV: udp_received_data.csv
"""

import socket
import struct
import csv
import os
from datetime import datetime


# ============================================================
# UDP CONFIGURATION
# ============================================================
UDP_HOST = "0.0.0.0"
UDP_PORT = 5005
BUFFER_SIZE = 4096

# Packet format:
# <ffffi = little endian, 4 float values, 1 integer value
PACKET_FORMAT = "<ffffi"
RECORD_SIZE = struct.calcsize(PACKET_FORMAT)

OUTPUT_CSV = "udp_received_data.csv"


# ============================================================
# CSV HEADER
# ============================================================
CSV_COLUMNS = [
    "Timestamp",
    "Time",
    "Height",
    "Resultant_acceleration",
    "Resultant_velocity",
    "AGC"
]


def create_csv_if_not_exists(csv_path):
    """Create CSV file with header if it does not exist."""
    if not os.path.exists(csv_path):
        with open(csv_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(CSV_COLUMNS)


def decode_dat_packet(raw_data):
    """
    Decode UDP .dat binary packet into structured records.

    Expected each record size = 20 bytes:
    Time                   -> float
    Height                 -> float
    Resultant_acceleration -> float
    Resultant_velocity     -> float
    AGC                    -> int
    """
    records = []

    for i in range(0, len(raw_data), RECORD_SIZE):
        chunk = raw_data[i:i + RECORD_SIZE]

        if len(chunk) == RECORD_SIZE:
            try:
                time_val, height, acceleration, velocity, agc = struct.unpack(
                    PACKET_FORMAT,
                    chunk
                )

                record = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "Time": round(float(time_val), 4),
                    "Height": round(float(height), 4),
                    "Resultant_acceleration": round(float(acceleration), 4),
                    "Resultant_velocity": round(float(velocity), 4),
                    "AGC": int(agc)
                }

                records.append(record)

            except struct.error as e:
                print(f"Packet decoding error: {e}")

    return records


def append_records_to_csv(records, csv_path):
    """Append decoded records into CSV file."""
    if not records:
        return

    with open(csv_path, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)

        for record in records:
            writer.writerow(record)


def start_udp_receiver():
    """
    Start UDP receiver.
    Receives .dat packets, decodes them, and saves them into CSV.
    """
    create_csv_if_not_exists(OUTPUT_CSV)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((UDP_HOST, UDP_PORT))

    print("==============================================")
    print(" UDP .dat Receiver Started")
    print("==============================================")
    print(f"Listening on     : {UDP_HOST}:{UDP_PORT}")
    print(f"Packet Format    : {PACKET_FORMAT}")
    print(f"Record Size      : {RECORD_SIZE} bytes")
    print(f"Saving CSV file  : {OUTPUT_CSV}")
    print("Press CTRL + C to stop")
    print("==============================================")

    try:
        while True:
            raw_data, sender_address = sock.recvfrom(BUFFER_SIZE)

            decoded_records = decode_dat_packet(raw_data)

            if decoded_records:
                append_records_to_csv(decoded_records, OUTPUT_CSV)

                print(
                    f"Received {len(decoded_records)} record(s) "
                    f"from {sender_address} -> saved to CSV"
                )

            else:
                print("Received packet but no valid record decoded.")

    except KeyboardInterrupt:
        print("\nUDP receiver stopped by user.")

    except Exception as e:
        print(f"UDP receiver error: {e}")

    finally:
        sock.close()
        print("Socket closed.")


if __name__ == "__main__":
    start_udp_receiver()
