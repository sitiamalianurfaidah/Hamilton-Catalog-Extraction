CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE construction.items_specification (
    id                  SERIAL PRIMARY KEY,
    result_json_path    TEXT NOT NULL,
    nama_proyek         TEXT,
    nama_site           TEXT,
    lokasi              TEXT,
    nama_bangunan       TEXT,
    jenis_pekerjaan     TEXT,
    header1             TEXT,
    header2             TEXT,
    nama_barang         TEXT NOT NULL,
    spesifikasi_general TEXT,
    spesifikasi_merek   TEXT,
    search_text         TEXT NOT NULL,
    embedding           vector(1024),
    created_at          TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX ON construction.items_specification USING ivfflat (embedding vector_cosine_ops);
