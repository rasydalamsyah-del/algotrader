import os

def gabungkan_semua_ke_txt(output_file="full_project_context.txt"):
    # Folder yang sebaiknya diabaikan agar file hasil tidak kotor
    folder_blacklist = {'.git', '__pycache__', '.vscode', 'venv', 'env', 'node_modules'}
    # File hasil itu sendiri jangan ikut dibaca (recursive loop)
    file_blacklist = {output_file, 'gabung_file.py'} 
    
    direktori_saat_ini = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Sedang merangkum proyek di: {direktori_saat_ini}...")
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for root, dirs, files in os.walk(direktori_saat_ini):
            # Modifikasi dirs di tempat untuk skip folder blacklist
            dirs[:] = [d for d in dirs if d not in folder_blacklist]
            
            for nama_file in files:
                if nama_file in file_blacklist:
                    continue
                
                jalur_lengkap = os.path.join(root, nama_file)
                # Dapatkan jalur relatif untuk header agar struktur folder terlihat
                jalur_relatif = os.path.relpath(jalur_lengkap, direktori_saat_ini)
                
                f_out.write(f"--- START OF FILE: {jalur_relatif} ---\n")
                
                try:
                    # Coba baca sebagai teks
                    with open(jalur_lengkap, 'r', encoding='utf-8') as f_in:
                        # Kita intip dikit isinya, kalau ada karakter null (\x00) biasanya itu binary
                        content = f_in.read()
                        if '\x00' in content:
                            f_out.write("[SKIP: File terdeteksi sebagai binary/bukan teks]\n")
                        else:
                            f_out.write(content)
                except UnicodeDecodeError:
                    f_out.write("[SKIP: Gagal membaca (kemungkinan file binary atau encoding berbeda)]\n")
                except Exception as e:
                    f_out.write(f"[ERROR: Tidak bisa membaca file karena {e}]\n")
                
                f_out.write(f"\n--- END OF FILE: {jalur_relatif} ---\n\n")
                    
    print(f"Selesai! Semua isi file proyek lu udah ada di: {output_file}")

if __name__ == "__main__":
    gabungkan_semua_ke_txt()
