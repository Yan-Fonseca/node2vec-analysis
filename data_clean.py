import sys


def remove_last_column(file_path: str) -> None:
    """Reescreve o arquivo removendo a última coluna de cada linha."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cleaned_lines = []
    for line in lines:
        stripped = line.rstrip("\n")
        if not stripped:
            cleaned_lines.append("\n")
            continue

        parts = stripped.split()
        if len(parts) <= 1:
            cleaned_lines.append("\n")
            continue

        cleaned_lines.append(" ".join(parts[:-1]) + "\n")

    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(cleaned_lines)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python data_clean.py <arquivo>")
        sys.exit(1)

    remove_last_column(sys.argv[1])
