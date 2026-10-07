# R48: Benchmark Corpus Investigation — Silesia Morphology & Modern Alternatives

**Date**: 2026-10-07
**Status**: Complete
**Purpose**: Investigate whether the Silesia corpus (2003) remains adequate for cross-domain validation, identify modern alternatives, and design a Tier 3 evaluation suite with current data morphologies.

## Background

Silesia was created in 2003 to replace Calgary (1987) and Canterbury (1997). Its 12 files (211 MB) represent data types of that era. After 23 years, the question is whether these files still represent modern data morphologies for compression benchmarking.

## Finding 1: Silesia IS morphologically outdated

### Files with anachronistic morphology

| Silesia file | What it is (2003) | Modern equivalent (2026) | Morphology change |
|---|---|---|---|
| mozilla | x86-32 ELF, Mozilla ~2003 | x86-64 PE/ELF with PIE/ASLR, or ARM64, or WASM | Instruction encoding, section layout, linking model |
| ooffice | OpenOffice 1.01 binary | Modern Office365/LibreOffice (OOXML = ZIP+XML) | Completely different format |
| osdb | MySQL 3.23 database | Parquet, Arrow, SQLite WAL, PostgreSQL pages | Columnar vs row-oriented |
| samba | Samba C source ~2003 | Modern Rust/Go/TypeScript source, or JSON configs | Language structure, naming conventions |

### Data types completely absent from Silesia

- **JSON/API data**: dominant structured format since ~2010
- **Container images**: Docker layers, OCI format
- **ML model weights**: SafeTensors, GGUF, PyTorch .pt
- **WebAssembly**: modern bytecode format (4-byte aligned, no E8/E9)
- **Structured logs**: JSON-lines, syslog, CloudWatch
- **Cloud-native formats**: Parquet, Arrow, Protobuf, MessagePack
- **Modern media metadata**: HEIF, AV1 OBU, VP9 IVF

### Squash Corpus criticism (2015)

The Squash Corpus project explicitly aimed to replace Silesia, noting: "existing corpora for data compression are not an accurate representation of the types of things people typically use compression for. Most people probably don't have many large plain text files (like books), but most existing corpora are rife with them." The project listed JSON, office documents, PDFs, logs, and game data as "woefully underrepresented." Status: abandoned (WIP since 2015).

## Finding 2: No consolidated replacement exists

| Benchmark | Year | Files | Scope | Status |
|---|---|---|---|---|
| Silesia | 2003 | 12 (211 MB) | General-purpose | Active by inertia |
| enwik8/9 (LTCB) | 2006 | 1 (100MB/1GB) | Text (Wikipedia) | Active (Hutter Prize) |
| **AIT DCC 2026** | 2026 | 16+4 (A-T) | Hidden test + multi-domain | **Most relevant modern** |
| Squash Corpus | 2015 | 9 (756 MB) | Real-world data | Abandoned |
| AstroCompress | 2025 | 5 datasets | Astronomical imaging | Domain-specific |
| FCBench | 2023 | Multi | Floating-point lossless | Domain-specific |
| SDRBench | 2021 | Multi | Scientific lossy | Domain-specific |
| TSCom-Bench | 2025 | Multi | Time series | Domain-specific |

### AIT DCC 2026 — the most significant modern effort

The 2026 Algorithmic Information Theory Data Compression Challenge is the first serious modern general-purpose compression benchmark:

- **16 heterogeneous files** (A-P): protein sequences, C source, Wikipedia text, pseudo-random, CERN ATLAS float, astronomical images, executable binary
- **8 training + 8 hidden testing** — prevents corpus-specific overfitting
- **4 external validation files** (Q-T): human genome, DBLP XML, Caltech-256 JPEG TAR, OpenStreetMap PBF
- **117 compressors evaluated** including gzip, zstd, lzma, PAQ8px
- **Key finding**: "the exact Pareto-optimal set changes between training and testing panels"
- **Publicly available**: all A-P files downloadable from aitdcc.github.io
- **Total size**: ~38 MB (A-P combined)

### Hutter Prize 2026 state

- **Vladimir Ivanov** (July 2026): fx2-cmix-T → 100,424,672 bytes (9.96x), €37,300
- **David Freelan** (July 2026): cmix-obias → 108,521,870 bytes (9.21x), €5,240
- Progress: ~1% per year over 20 years. Current goal: enwik9 < 100MB.
- Our target (sub-1.0 BPB on enwik8) is aligned with this competition.

## Finding 3: E8/E9 transform is x86-specific

Our E8/E9 preprocessing (N3) targets x86 CALL/JMP opcodes. This has architectural implications:

| Architecture | Filter name | Target opcodes | Displacement | Improvement | Maturity |
|---|---|---|---|---|---|
| x86-32 | BCJ/E8E9 | CALL(0xE8), JMP(0xE9) | 32-bit relative | 6-8% | Since 1996 |
| x86-64 | BCJ/E8E9 | CALL(0xE8), JMP(0xE9) | 32-bit relative | 6-8% | Same filter works |
| ARM64 | BCJ-ARM64 | BL(0x94xx), ADRP(0x90xx) | 26-bit word-scaled | ~5% | Since 2022 (xz 5.4+) |
| WASM | N/A | No relative branches | N/A | 0% | Not applicable |

E8/E9 remains valid for x86-64 (the opcode format is identical), but does NOT cover ARM64 or WASM. For a truly universal compressor, BCJ-ARM64 support would be needed in the future.

## Finding 4: Silesia should be COMPLEMENTED, not replaced

**Why keep Silesia:**
1. **Comparability**: the entire ecosystem reports on Silesia. PAQ8px, cmix, Nacrith, zstd — all have Silesia numbers. Without comparable numbers, no positioning.
2. **Data types are stable**: an x86 executable from 2003 and 2026 share fundamental structure (opcodes, relocations, sections). A medical image is a medical image.
3. **AIT 2026 confirms the categories**: even their modern benchmark includes executables, source code, text, scientific data — the same categories as Silesia.

**Why complement with modern data:**
1. **Missing morphologies**: JSON, WASM, ML weights, structured logs are absent.
2. **Overfitting risk**: optimizing for 12 specific files from 2003 may not generalize.
3. **AIT 2026 showed**: hidden testing changes the Pareto-optimal set of compressors.

## Recommendation

Create a **Tier 3 (T3) Modern Morphology** evaluation suite:
- Does NOT replace T1 (5 files) or T2 (12 Silesia files)
- Adds modern data types absent from Silesia
- Uses AIT DCC 2026 files (publicly available, peer-reviewed, multi-domain)
- Supplements with local modern data (our own x86-64 PE binary, SafeTensors weights, JSONL logs)
- Smoke-test level: 10KB per file, same methodology as T1/T2

See `docs/BENCHMARKS.md` for T3 specification.

## References

- [2026 AIT Data Compression Challenge](https://arxiv.org/html/2606.17712v1)
- [AIT DCC Dataset](https://aitdcc.github.io/dataset.html)
- [Silesia Compression Corpus](https://sun.aei.polsl.pl/~sdeor/index.php?page=silesia)
- [Silesia Benchmark (Matt Mahoney)](http://mattmahoney.net/dc///silesia.html)
- [Squash Corpus Project](https://github.com/nemequ/squash-corpus)
- [BCJ Algorithm — Wikipedia](https://en.wikipedia.org/wiki/BCJ_(algorithm))
- [Hutter Prize](https://hutter1.net/prize/)
- [Large Text Compression Benchmark](https://mattmahoney.net/dc/text.html)
- [AstroCompress](https://arxiv.org/html/2506.08306v1)
- [FCBench](https://arxiv.org/pdf/2312.10301)
