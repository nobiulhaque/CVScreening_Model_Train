"""
Create unified dataset from ALL data sources inside dataset/ directory.

Fully dynamic — categories are discovered automatically:
  1. Folder names = categories (dataset/SomeCategory/*.pdf → "somecategory")
  2. CSV column values = categories (category/position column values used as-is)

No hardcoded category list. Drop resumes into any folder and run this.
Outputs unified feature CSVs + JSON into processed_data/.
"""

import csv
import json
import ast
import hashlib
import os
import re
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from feature_extractor import ResumeFeatureExtractor, extract_text_from_file
from config import (
    DATASET_DIR, PROCESSED_DATA_DIR,
    TECHNICAL_SKILLS, SOFT_SKILLS, SUPPORTED_EXTENSIONS,
)


# ======================================================================
# Helpers
# ======================================================================

def normalize_category(name: str) -> str:
    """Standardize category names to merge duplicates."""
    name = name.lower().strip()
    # 1. Strip common noise suffixes like '-resumes'
    name = re.sub(r'[-_\s]*(resumes?|cv|profiler?)$', '', name)
    # 2. Merge hyphenated/spaced variations (data-science vs datascience)
    name = re.sub(r'[^a-z0-9]', '', name)
    
    # 3. Handle manual merges for synonyms and near-duplicates
    mapping = {
        "advocate": "legal",
        "advocateresumes": "legal",
        "civil": "civil-engineering",
        "civilengineer": "civil-engineering",
        "civilengineering": "civil-engineering",
        "hr": "human-resources",
        "humanresources": "human-resources",
        "managment": "management",
        "operationmanager": "operations",
        "operationsmanager": "operations",
        "python": "python-developer",
        "pythondeveloper": "python-developer",
        "java": "java-developer",
        "javadeveloper": "java-developer",
        "agricultural": "agriculture",
        "electricalengineer": "electrical-engineering",
        "electricalengineering": "electrical-engineering",
        "informationtechnology": "it",
        "itresumes": "it",
        "sqldeveloper": "sql",
        "etldeveloper": "etl",
        "reactdeveloper": "react",
        "dotnetdeveloper": "dotnet",
        "webdesigning": "design",
        "designing": "design",
        "designer": "design",
        "pmo": "project-management",
        "pbo": "project-management",
    }
    return mapping.get(name, name)


def text_hash(text: str) -> str:
    """Quick content hash for deduplication."""
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()


def safe_parse_list(value: str) -> list:
    """Parse a Python-style list string like \"['a','b']\" into a real list."""
    if not value or value.strip() in ("", "[]", "None", "N/A"):
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(x) for x in parsed if x is not None]
        return [str(parsed)]
    except Exception:
        return [value.strip()]


# ======================================================================
# Source discovery
# ======================================================================

def discover_resume_folders(root: Path):
    """Recursively find folders that contain resume files.
    Each folder's name becomes its category (fully dynamic).

    Returns list of (folder_path, category_name).
    """
    results = []

    for dirpath in sorted(root.glob("**/")):
        if dirpath == root:
            continue
        # A folder is a "resume folder" if it directly contains resume files
        has_files = any(
            f.suffix.lower() in SUPPORTED_EXTENSIONS
            for f in dirpath.iterdir() if f.is_file()
        )
        if not has_files:
            continue

        category = normalize_category(dirpath.name)
        results.append((dirpath, category))

    return results


def discover_csv_files(root: Path):
    """Find all CSV files anywhere under root."""
    return sorted(root.rglob("*.csv"))


# ======================================================================
# Multiprocessing Workers
# ======================================================================

def _process_file_worker(file_path, category, source):
    """Worker function to process a single file in parallel."""
    try:
        # Initialize extractor inside worker to avoid pickle issues
        from feature_extractor import ResumeFeatureExtractor, extract_text_from_file
        extractor = ResumeFeatureExtractor()
        
        text = extract_text_from_file(file_path)
        if not text or len(text) < 50:
            return "skipped_short", None, None

        h = hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()
        
        features = extractor.extract_all(text)
        features["category"] = category
        features["filename"] = file_path.name
        features["source"] = source
        
        return "success", h, features
    except Exception as e:
        return "error", str(e), file_path.name


# ======================================================================
# Processing
# ======================================================================

def process_resume_folders(folders, extractor, seen_hashes):
    """Extract features from raw resume files in discovered folders in PARALLEL.
    Deduplicates by content hash.
    """
    all_data = []
    stats = Counter()

    format_priority = {
        ".pdf": 0, ".docx": 1, ".doc": 2, ".txt": 3,
        ".png": 4, ".jpg": 4, ".jpeg": 4, ".bmp": 4, ".tiff": 4,
    }

    # Step 1: Collect all work items
    work_items = []
    for folder_path, category in folders:
        files = [
            f for f in sorted(folder_path.iterdir())
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        # Deduplicate by stem (skip .txt if .pdf or .docx exists)
        seen_stems = set()
        for f in sorted(files, key=lambda x: (x.stem.lower(), format_priority.get(x.suffix.lower(), 99), x.suffix.lower())):
            if f.stem not in seen_stems:
                seen_stems.add(f.stem)
                try:
                    source = str(folder_path.relative_to(folder_path.parents[2]))
                except (IndexError, ValueError):
                    source = folder_path.name
                work_items.append((f, category, source))

    # Step 2: Use multiprocessing pool
    print(f"\n  [Multiprocessing] Using {os.cpu_count()} CPU cores ...")
    
    with tqdm(total=len(work_items), desc="Parsing Resumes", unit="file", leave=True) as pbar:
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Submit all jobs
            future_to_file = {
                executor.submit(_process_file_worker, f, cat, src): f
                for f, cat, src in work_items
            }
            
            for future in as_completed(future_to_file):
                status, result, content = future.result()
                
                if status == "success":
                    h = result # hash
                    features = content # features dict
                    
                    if h in seen_hashes:
                        stats["skipped_dup"] += 1
                    else:
                        seen_hashes.add(h)
                        all_data.append(features)
                        stats[features["category"]] += 1
                elif status == "skipped_short":
                    stats["skipped_short"] += 1
                elif status == "error":
                    print(f"    [SKIP] Error processing {content}: {result}")
                    stats["errors"] += 1
                
                pbar.update(1)

    print(f"\n  Raw resume files processed:")
    for cat in sorted(c for c in stats if c not in ("skipped_dup", "skipped_short", "errors")):
        print(f"    {cat:20} {stats.get(cat, 0):5}")
    print(f"    {'duplicates skipped':20} {stats.get('skipped_dup', 0):5}")
    print(f"    {'too short skipped':20} {stats.get('skipped_short', 0):5}")
    print(f"    {'errors':20} {stats.get('errors', 0):5}")

    return all_data


def _csv_row_to_text(row):
    """Reconstruct a pseudo-resume from a structured CSV row."""
    parts = []

    # Career objective
    obj = row.get("career_objective", "").strip()
    if obj:
        parts.append("Career Objective: " + obj)

    # Skills
    skills = safe_parse_list(row.get("skills", ""))
    if skills:
        parts.append("Skills: " + ", ".join(skills))

    # Education
    degrees = safe_parse_list(row.get("degree_names", ""))
    institutions = safe_parse_list(row.get("educational_institution_name", ""))
    fields = safe_parse_list(row.get("major_field_of_studies", ""))
    years = safe_parse_list(row.get("passing_years", ""))
    if degrees or institutions:
        edu_parts = []
        for i in range(max(len(degrees), len(institutions))):
            d = degrees[i] if i < len(degrees) else ""
            inst = institutions[i] if i < len(institutions) else ""
            field = fields[i] if i < len(fields) else ""
            yr = years[i] if i < len(years) else ""
            edu_parts.append(("{} {} from {} ({})".format(d, field, inst, yr)).strip())
        parts.append("Education: " + "; ".join(edu_parts))

    # Experience
    companies = safe_parse_list(row.get("professional_company_names", ""))
    positions = safe_parse_list(row.get("positions", ""))
    start_dates = safe_parse_list(row.get("start_dates", ""))
    end_dates = safe_parse_list(row.get("end_dates", ""))
    for i in range(max(len(companies), len(positions))):
        pos = positions[i] if i < len(positions) else ""
        comp = companies[i] if i < len(companies) else ""
        sd = start_dates[i] if i < len(start_dates) else ""
        ed = end_dates[i] if i < len(end_dates) else ""
        parts.append("Experience: {} at {} ({} - {})".format(pos, comp, sd, ed))

    # Responsibilities
    resp = row.get("responsibilities", "").strip()
    if resp:
        parts.append("Responsibilities: " + resp)

    # Certifications
    certs = safe_parse_list(row.get("certification_skills", ""))
    if certs:
        parts.append("Certifications: " + ", ".join(certs))

    # Languages
    langs = safe_parse_list(row.get("languages", ""))
    if langs:
        parts.append("Languages: " + ", ".join(langs))

    return "\n".join(parts)


def process_csv_files(csv_paths, extractor, seen_hashes):
    """Process CSV resume datasets. Each row is converted to pseudo-resume
    text, feature-extracted, and mapped to an internal category.
    """
    all_data = []
    stats = Counter()

    for csv_path in csv_paths:
        print(f"\n  Processing CSV: {csv_path.name}")
        try:
            with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []

                # Detect which column holds the category.
                # Priority: explicit category columns > job position > raw positions list
                position_col = None
                for candidate in ["category", "Category", "Resume_Category",
                                  "Resume_category", "job_category",
                                  "\ufeffjob_position_name", "job_position_name",
                                  "job_title", "Job_Title", "position",
                                  "Position", "positions"]:
                    if candidate in fieldnames:
                        position_col = candidate
                        break

                if position_col is None:
                    print(f"    WARNING: No position/category column found in {csv_path.name}, skipping")
                    continue

                # Detect if there is a raw resume_str / Resume_str text column
                text_col = None
                for candidate in ["resume_str", "Resume_str", "resume_text",
                                  "Resume", "resume", "text", "resume_html"]:
                    if candidate in fieldnames:
                        text_col = candidate
                        break

                row_count = 0
                for row in reader:
                    row_count += 1
                    # Determine category directly from CSV value
                    position_raw = row.get(position_col, "").strip()
                    if not position_raw:
                        stats["empty_category"] += 1
                        continue
                    category = normalize_category(position_raw)

                    # Build text
                    if text_col and row.get(text_col, "").strip():
                        text = row[text_col].strip()
                    else:
                        text = _csv_row_to_text(row)

                    if len(text) < 50:
                        stats["skipped_short"] += 1
                        continue

                    h = text_hash(text)
                    if h in seen_hashes:
                        stats["skipped_dup"] += 1
                        continue
                    seen_hashes.add(h)

                    features = extractor.extract_all(text)
                    features["category"] = category
                    features["filename"] = "{}_row{}".format(csv_path.stem, row_count)
                    features["source"] = csv_path.name
                    all_data.append(features)
                    stats[category] += 1

        except Exception as e:
            print(f"    ERROR processing {csv_path.name}: {e}")
            stats["csv_errors"] += 1

    print(f"\n  CSV data processed:")
    skip_keys = {"empty_category", "skipped_dup", "skipped_short", "csv_errors"}
    for cat in sorted(c for c in stats if c not in skip_keys):
        print(f"    {cat:20} {stats.get(cat, 0):5}")
    print(f"    {'empty category':20} {stats.get('empty_category', 0):5}")
    print(f"    {'duplicates skipped':20} {stats.get('skipped_dup', 0):5}")
    print(f"    {'too short skipped':20} {stats.get('skipped_short', 0):5}")

    return all_data


# ======================================================================
# Output writers (same format as before - train.py compatible)
# ======================================================================

def write_outputs(all_data, categories):
    """Write the unified dataset to processed_data/ in all formats."""

    all_skills = TECHNICAL_SKILLS + SOFT_SKILLS

    # ---------- CSV 1: Main human-readable dataset ----------
    csv_path = PROCESSED_DATA_DIR / "resume_dataset.csv"
    columns = [
        "filename", "category", "source",
        "word_count", "text_length",
        "years_of_experience",
        "education_level", "education_score",
        "num_technical_skills", "num_soft_skills", "total_skills",
        "num_certifications",
        "has_email", "has_phone",
        "num_sections",
        "seniority_level", "seniority_score",
        "resume_quality_score",
        "num_employment_gaps", "total_gap_months",
        "num_red_flags",
        "candidate_name",
        "technical_skills_list", "soft_skills_list",
        "certifications_list", "job_titles_list",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for item in all_data:
            writer.writerow({
                "filename": item["filename"],
                "category": item["category"],
                "source": item.get("source", ""),
                "word_count": item["word_count"],
                "text_length": item["text_length"],
                "years_of_experience": item["years_of_experience"],
                "education_level": item["education_level"],
                "education_score": item["education_score"],
                "num_technical_skills": item["num_technical_skills"],
                "num_soft_skills": item["num_soft_skills"],
                "total_skills": item["total_skills"],
                "num_certifications": item["num_certifications"],
                "has_email": int(item["has_email"]),
                "has_phone": int(item["has_phone"]),
                "num_sections": item["num_sections"],
                "seniority_level": item.get("seniority_level", "unknown"),
                "seniority_score": item.get("seniority_score", 0),
                "resume_quality_score": item.get("resume_quality_score", 0),
                "num_employment_gaps": item.get("num_employment_gaps", 0),
                "total_gap_months": item.get("total_gap_months", 0),
                "num_red_flags": item.get("num_red_flags", 0),
                "candidate_name": item.get("candidate_name", ""),
                "technical_skills_list": "|".join(item["technical_skills"]),
                "soft_skills_list": "|".join(item["soft_skills"]),
                "certifications_list": "|".join(item["certifications"]),
                "job_titles_list": "|".join(item["job_titles"]),
            })
    print(f"  Main dataset:  {csv_path}  ({len(all_data)} rows)")

    # ---------- CSV 2: Skill vectors (one-hot) - used by train.py ----------
    skill_csv_path = PROCESSED_DATA_DIR / "resume_skill_vectors.csv"
    skill_columns = (
        ["filename", "category"]
        + ["skill_{}".format(s.replace(' ', '_')) for s in all_skills]
        + ["years_of_experience", "education_score",
           "num_certifications", "num_sections",
           "seniority_score", "resume_quality_score",
           "num_employment_gaps", "total_gap_months", "num_red_flags"]
    )
    with open(skill_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=skill_columns)
        writer.writeheader()
        for item in all_data:
            row = {
                "filename": item["filename"],
                "category": item["category"],
                "years_of_experience": item["years_of_experience"],
                "education_score": item["education_score"],
                "num_certifications": item["num_certifications"],
                "num_sections": item["num_sections"],
                "seniority_score": item.get("seniority_score", 0),
                "resume_quality_score": item.get("resume_quality_score", 0),
                "num_employment_gaps": item.get("num_employment_gaps", 0),
                "total_gap_months": item.get("total_gap_months", 0),
                "num_red_flags": item.get("num_red_flags", 0),
            }
            found_skills = item["technical_skills"] + item["soft_skills"]
            for skill in all_skills:
                row["skill_{}".format(skill.replace(' ', '_'))] = 1 if skill in found_skills else 0
            writer.writerow(row)
    print(f"  Skill vectors: {skill_csv_path}  ({len(skill_columns)} cols)")

    # ---------- JSON: Full features (minus raw text) ----------
    json_path = PROCESSED_DATA_DIR / "resume_features.json"
    json_data = [
        {k: v for k, v in item.items() if k != "raw_text"}
        for item in all_data
    ]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"  Features JSON: {json_path}")

    # ---------- Raw text corpus (for TF-IDF in train.py) ----------
    corpus_path = PROCESSED_DATA_DIR / "resume_raw_texts.json"
    corpus_data = [
        {"filename": item["filename"],
         "category": item["category"],
         "raw_text": item.get("raw_text", "")}
        for item in all_data
    ]
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(corpus_data, f, indent=2)
    print(f"  Raw text:      {corpus_path}")

    # ---------- Categories metadata (used by train.py) ----------
    cat_meta_path = PROCESSED_DATA_DIR / "categories.json"
    cat_counts = Counter(item["category"] for item in all_data)
    cat_meta = {
        "categories": categories,
        "counts": {cat: cat_counts.get(cat, 0) for cat in categories},
        "total_samples": len(all_data),
    }
    with open(cat_meta_path, "w", encoding="utf-8") as f:
        json.dump(cat_meta, f, indent=2)
    print(f"  Categories:    {cat_meta_path}  ({len(categories)} categories)")


# ======================================================================
# Main
# ======================================================================

def create_dataset():
    """Discover ALL data inside dataset/, process, and write unified dataset."""
    print("=" * 70)
    print("  UNIFIED DATASET BUILDER")
    print("  Scans everything inside: dataset/")
    print("=" * 70)

    if not DATASET_DIR.exists():
        print("ERROR: Dataset directory not found: {}".format(DATASET_DIR))
        return

    extractor = ResumeFeatureExtractor()
    seen_hashes = set()  # content-level dedup across ALL sources

    # Step 1: Discover resume folders
    print("\n[1/4] Discovering resume folders ...")
    folders = discover_resume_folders(DATASET_DIR)
    print(f"  Found {len(folders)} categorized folders")
    for folder, cat in folders:
        file_count = sum(1 for f in folder.iterdir()
                         if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS)
        rel = str(folder.relative_to(DATASET_DIR))
        print(f"    {rel:50} -> {cat:15} ({file_count} files)")

    # Step 2: Discover CSV files
    print("\n[2/4] Discovering CSV files ...")
    csv_files = discover_csv_files(DATASET_DIR)
    print(f"  Found {len(csv_files)} CSV file(s)")
    for cp in csv_files:
        print(f"    {cp.relative_to(DATASET_DIR)}")

    # Step 3: Process everything
    print("\n[3/4] Processing raw resume files ...")
    file_data = process_resume_folders(folders, extractor, seen_hashes)

    print("\n[3/4] Processing CSV files ...")
    csv_data = process_csv_files(csv_files, extractor, seen_hashes)

    all_data = file_data + csv_data

    if not all_data:
        print("\nERROR: No data extracted! Check your dataset/ directory.")
        return

    # Discover all unique categories from the data
    categories = sorted(set(item["category"] for item in all_data))
    print(f"\n  Discovered {len(categories)} categories: {categories}")

    # Step 4: Write outputs
    print(f"\n[4/4] Writing unified dataset ({len(all_data)} total samples) ...")
    write_outputs(all_data, categories)

    # ==================== Summary ====================
    print("\n" + "=" * 70)
    print("  DATASET SUMMARY")
    print("=" * 70)

    cat_counts = Counter(item["category"] for item in all_data)
    for cat in categories:
        count = cat_counts.get(cat, 0)
        pct = count / len(all_data) * 100
        bar = "#" * int(pct / 2)
        print(f"  {cat:20} {count:5} resumes  ({pct:5.1f}%)  {bar}")
    print(f"  {'TOTAL':20} {len(all_data):5} resumes")

    # Source breakdown
    source_counts = Counter(item.get("source", "unknown") for item in all_data)
    print(f"\n  Data sources:")
    for src, cnt in source_counts.most_common():
        print(f"    {src:40} {cnt:5} samples")

    # Feature stats
    print(f"\n  Avg years of experience: {sum(d['years_of_experience'] for d in all_data)/len(all_data):.1f}")
    print(f"  Avg technical skills:    {sum(d['num_technical_skills'] for d in all_data)/len(all_data):.1f}")
    print(f"  Avg soft skills:         {sum(d['num_soft_skills'] for d in all_data)/len(all_data):.1f}")
    print(f"  Avg education score:     {sum(d['education_score'] for d in all_data)/len(all_data):.1f}")

    edu_counts = Counter(item["education_level"] for item in all_data)
    print(f"\n  Education distribution:")
    for edu, count in sorted(edu_counts.items(), key=lambda x: -x[1]):
        print(f"    {edu:15} {count:5}")

    # Class balance warning
    max_cat = max(cat_counts.values())
    min_cat = min(cat_counts.values()) if cat_counts else 0
    if max_cat > 0 and min_cat / max_cat < 0.3:
        print(f"\n  WARNING: Class imbalance detected (ratio {min_cat/max_cat:.2f}).")
        print(f"     SMOTE will handle this during training.")

    print("=" * 70)


if __name__ == "__main__":
    create_dataset()
