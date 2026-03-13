import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import re

def excel_to_items_specification(excel_path: str, output_dir: str = "output") -> Dict[str, Any]:
    """
    Parse Excel file ke format items_specification.
    Khusus untuk format Monitoring PPM seperti contoh.
    """
    print(f"Reading Excel file: {excel_path}")
    
    # Baca semua sheet
    xl = pd.ExcelFile(excel_path)
    print(f"Sheets found: {xl.sheet_names}")
    
    # Ambil sheet pertama (Monitoring PPM) sebagai sumber data utama
    df = pd.read_excel(excel_path, sheet_name=0, header=None)
    
    # Ekstrak metadata proyek dari judul
    project_name = None
    for idx, row in df.iterrows():
        if idx < 5:  # Cek 5 baris pertama
            row_text = ' '.join([str(x) for x in row if pd.notna(x)])
            if 'MONITORING PPM' in row_text:
                # Extract project name
                match = re.search(r'MONITORING PPM\s+(.+)', row_text)
                if match:
                    project_name = match.group(1).strip()
                break
    
    # Cari baris header (dimana ada NO., NO. APPROVAL, dll)
    header_row_idx = None
    for idx, row in df.iterrows():
        row_values = [str(x).strip() if pd.notna(x) else '' for x in row]
        if 'NO.' in row_values and 'NAMA MATERIAL' in str(row_values):
            header_row_idx = idx
            break
    
    if header_row_idx is None:
        print("Warning: Could not find header row, using default")
        header_row_idx = 3  # Asumsi baris ke-4 adalah header
    
    # Set header
    df.columns = df.iloc[header_row_idx]
    df = df.iloc[header_row_idx + 1:].reset_index(drop=True)
    
    # Bersihkan nama kolom
    df.columns = [str(col).strip() if pd.notna(col) else f'Unnamed_{i}' 
                  for i, col in enumerate(df.columns)]
    
    print(f"Columns found: {list(df.columns)}")
    
    # Identifikasi kolom-kolom penting
    col_mapping = {
        'nama_barang': None,
        'spesifikasi': None,
        'merk': None,
        'pekerjaan': None
    }
    
    for col in df.columns:
        col_lower = str(col).lower()
        if 'nama material' in col_lower or 'material' in col_lower:
            col_mapping['nama_barang'] = col
        elif 'spesifikasi' in col_lower:
            col_mapping['spesifikasi'] = col
        elif 'merk' in col_lower:
            col_mapping['merk'] = col
        elif 'pekerjaan' in col_lower:
            col_mapping['pekerjaan'] = col
    
    print(f"Column mapping: {col_mapping}")
    
    # Proses setiap baris untuk membuat items
    items = []
    current_header1 = None
    current_header2 = None
    
    for idx, row in df.iterrows():
        # Skip baris yang benar-benar kosong
        if row.isna().all():
            continue
        
        # Cek apakah ini baris header (kolom NO. APPROVAL kosong tapi ada nama material)
        no_approval = str(row.get('NO. APPROVAL', '')) if 'NO. APPROVAL' in df.columns else ''
        
        # Ambil nama barang
        nama_barang_col = col_mapping.get('nama_barang')
        if nama_barang_col and pd.notna(row.get(nama_barang_col)):
            nama_barang = str(row[nama_barang_col]).strip()
            
            # Skip jika hanya angka (seperti "1", "2", dll) - ini adalah nomor urut
            if nama_barang.isdigit() and len(nama_barang) < 4:
                continue
            
            # Ambil spesifikasi
            spesifikasi = ""
            if col_mapping.get('spesifikasi') and pd.notna(row.get(col_mapping['spesifikasi'])):
                spesifikasi = str(row[col_mapping['spesifikasi']]).strip()
                # Bersihkan HTML tags jika ada
                spesifikasi = re.sub(r'<[^>]+>', '', spesifikasi)
            
            # Ambil merk
            merk = ""
            if col_mapping.get('merk') and pd.notna(row.get(col_mapping['merk'])):
                merk = str(row[col_mapping['merk']]).strip()
                if merk.startswith('Ex.'):
                    merk = merk.replace('Ex.', '').strip()
            
            # Tentukan header berdasarkan pekerjaan
            if col_mapping.get('pekerjaan') and pd.notna(row.get(col_mapping['pekerjaan'])):
                pekerjaan = str(row[col_mapping['pekerjaan']]).strip()
                if pekerjaan and pekerjaan != 'nan':
                    current_header1 = pekerjaan
                    current_header2 = None
            
            # Deteksi sub-header dari baris sebelumnya yang mungkin kosong di kolom NO. APPROVAL
            if pd.isna(no_approval) or no_approval == 'nan' or no_approval == '':
                # Ini mungkin sub-header atau item lanjutan
                if nama_barang and not any(x in nama_barang.lower() for x in ['uk.', 'dia.', 'tebal']):
                    # Mungkin ini header2
                    if nama_barang and len(nama_barang) < 50:  # Hindari deskripsi panjang
                        current_header2 = nama_barang
                        continue  # Jangan tambah sebagai item
            
            # Buat item
            item = {
                "header1": current_header1,
                "header2": current_header2,
                "nama_barang": nama_barang,
                "spesifikasi_barang": {
                    "spesifikasi_general": spesifikasi,
                    "spesifikasi_merek": merk if merk else None
                }
            }
            
            # Tambahkan hanya jika nama_barang tidak kosong dan bukan header semata
            if nama_barang and nama_barang not in ['NO.', 'No'] and len(nama_barang) > 1:
                items.append(item)
                print(f"Added: {nama_barang[:30]}... (Header1: {current_header1}, Header2: {current_header2})")
    
    # Ekstrak metadata proyek
    metadata = {
        "nama_proyek": project_name or Path(excel_path).stem,
        "nama_site": None,
        "lokasi": None,
        "nama_bangunan": None,
        "jenis_pekerjaan": [],
        "items": items
    }
    
    print(f"\nTotal items extracted: {len(items)}")
    return metadata

def excel_to_result_json(excel_path: str, output_dir: str = None) -> str:
    """
    Konversi Excel ke result.json format.
    """
    import time
    from datetime import datetime
    
    if output_dir is None:
        timestamp = int(time.time())
        output_dir = f"output/{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse Excel
    print(f"\nProcessing Excel: {excel_path}")
    result = excel_to_items_specification(excel_path, output_dir)
    
    # Simpan sebagai result.json
    result_path = os.path.join(output_dir, "result.json")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\nExcel parsed: {len(result['items'])} items → {result_path}")
    
    # Tampilkan beberapa contoh item
    print("\nSample items:")
    for i, item in enumerate(result['items'][:5]):
        print(f"  {i+1}. {item['nama_barang']} - {item['spesifikasi_barang']['spesifikasi_general'][:30]}")
    
    return result_path

# Untuk testing langsung
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        excel_to_result_json(sys.argv[1])
    else:
        print("Usage: python excel_parser.py <excel_file>")