import pandas as pd
import re
import matplotlib.pyplot as plt


class CytoPy:
    def __init__(self):
        # Patterns for ISCN components
        self.clone_pattern = r"([^/]+)"
        self.meta_pattern = r"^(\d{2}~?\d{0,2}),([xy]+|x|y)?(.*?)\[(\d+|cp\d+)\]"

    def clean_string(self, text):
        """Removes quotes, whitespace, and non-ISCN descriptors."""
        if not isinstance(text, str):
            return ""
        text = text.lower().replace('"', "").strip()
        if "complex" in text or "keinezellen" in text:
            return "Non-Standard"
        return text

    def parse_iscn(self, iscn_str):
        """Parses an ISCN string into a list of clone dictionaries."""
        cleaned = self.clean_string(iscn_str)
        clones = re.findall(self.clone_pattern, cleaned)
        parsed_data = []

        for clone in clones:
            match = re.search(self.meta_pattern, clone)
            if match:
                parsed_data.append(
                    {
                        "chr_count": match.group(1),
                        "sex": match.group(2),
                        "abnormalities": match.group(3).strip(","),
                        "cell_count": match.group(4).replace("cp", ""),
                    }
                )
            else:
                # Handle cases without cell counts like "46,xx"
                simple_match = re.search(r"^(\d{2}),([xy]+)", clone)
                if simple_match:
                    parsed_data.append(
                        {
                            "chr_count": simple_match.group(1),
                            "sex": simple_match.group(2),
                            "abnormalities": "normal" if len(clone) < 10 else "unknown",
                            "cell_count": "1",  # Default if not specified
                        }
                    )
        return parsed_data

    def to_dataframe(self, iscn_list):
        """Converts a list of ISCN strings to a flattened pandas DataFrame."""
        rows = []
        for original in iscn_list:
            clones = self.parse_iscn(original)
            for i, clone in enumerate(clones):
                row = {"original_string": original, "clone_id": i + 1, **clone}
                rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            df["cell_count"] = pd.to_numeric(df["cell_count"], errors="coerce")
        return df

    def visualize_abnormalities(self, df):
        """Simple visualization of abnormality frequency."""
        if "abnormalities" not in df.columns:
            return

        # Filter out normal and empty
        subset = df[~df["abnormalities"].isin(["normal", ""])]
        counts = subset["chr_count"].value_counts()

        counts.plot(kind="bar", color="skyblue", edgecolor="black")
        plt.title("Distribution of Chromosome Counts in Abnormal Clones")
        plt.xlabel("Chromosome Count")
        plt.ylabel("Frequency")
        plt.show()


# --- Execution with Test Set ---
test_set = [
    "46,xy,del(20)(q12)[2]/46,xy[18]",
    "46,xx",
    "47,xy,+8[16]/46,xy[2]",
    "45,xx,del(5)(q13q33),inc[2]/46,xx[2]",
    "complex",
    "47,xx,der(2)t(2;11)(q37;p15),+8,r(11)[26]",
]

lib = CytoPy()
df_cytogenics = lib.to_dataframe(test_set)

print("### Processed Cytogenetics Data ###")
print(df_cytogenics[["chr_count", "sex", "abnormalities", "cell_count"]].head(10))
