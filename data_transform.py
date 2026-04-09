"""
data_transform.py — MODIFIABLE.
Feature engineering for blood cancer survival prediction.

This is the PRIMARY file for feature engineering experiments.
Both train.py and prepare.py import from this module.

The AutoFeatureBuilder discovers its vocabulary from training data:
which genes exist, which cytogenetic abnormalities are common, which
mutation effects appear, and which gene pairs co-mutate.

AI AGENT: This file is your main lever for improving the C-index.
Key areas for improvement:
  - CYTOGENETICS parsing: extract richer features from ISCN karyotype strings
  - Molecular features: gene-level, pathway-level, mutation burden
  - Clinical interactions: domain-specific ratios and combinations
  - New feature families: ELN risk groups, WHO classification proxies
  - Feature selection: smarter filtering beyond variance threshold
"""

import re
import numpy as np
import pandas as pd
from itertools import combinations


# =====================================================================
# AUTO FEATURE BUILDER
# =====================================================================
class AutoFeatureBuilder:
    """
    Automatically discovers feature vocabulary from training data and
    builds a comprehensive feature matrix for any patient set.

    Nothing is hardcoded — genes, cytogenetic tokens, mutation effects,
    co-mutation pairs, and chromosome identifiers are all learned from
    the data during fit().
    """

    CLIN_COLS = ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]

    def __init__(
        self,
        min_gene_patients=5,
        min_cyto_patients=8,
        min_effect_patients=5,
        n_top_comut_genes=20,
        min_comut_patients=8,
    ):
        self.min_gene_patients = min_gene_patients
        self.min_cyto_patients = min_cyto_patients
        self.min_effect_patients = min_effect_patients
        self.n_top_comut_genes = n_top_comut_genes
        self.min_comut_patients = min_comut_patients
        self._fitted = False

    # -----------------------------------------------------------------
    # FIT — learn vocabulary from training data
    # -----------------------------------------------------------------
    def fit(self, clinical_df, molecular_df):
        mol = molecular_df.copy()

        # Genes: keep those mutated in >= threshold patients
        gc = mol.groupby("GENE")["ID"].nunique().sort_values(ascending=False)
        self.genes_ = gc[gc >= self.min_gene_patients].index.tolist()

        # Mutation effects
        if "EFFECT" in mol.columns:
            ec = mol.groupby("EFFECT")["ID"].nunique()
            self.effects_ = ec[ec >= self.min_effect_patients].index.tolist()
        else:
            self.effects_ = []

        # Cytogenetic abnormality tokens
        self._fit_cyto_tokens(clinical_df)

        # Co-mutation gene pairs
        self._fit_comut_pairs(mol)

        # Chromosomes with mutations
        if "CHR" in mol.columns:
            mol["_chr"] = mol["CHR"].astype(str).str.strip()
            cc = mol.groupby("_chr")["ID"].nunique()
            self.chroms_ = cc[cc >= self.min_gene_patients].index.tolist()
        else:
            self.chroms_ = []

        self._fitted = True
        return self

    def _fit_cyto_tokens(self, cdf):
        counts = {}
        for s in cdf["CYTOGENETICS"].dropna():
            for tok in self._tokenize_cyto(str(s)):
                counts[tok] = counts.get(tok, 0) + 1
        self.cyto_tokens_ = sorted(
            tok for tok, c in counts.items() if c >= self.min_cyto_patients
        )

    def _fit_comut_pairs(self, mol):
        top = self.genes_[: self.n_top_comut_genes]
        sub = mol[mol["GENE"].isin(top)]
        if sub.empty:
            self.comut_pairs_ = []
            return
        piv = sub.groupby(["ID", "GENE"]).size().unstack(fill_value=0)
        piv = (piv > 0).astype(int)
        pairs = []
        for g1, g2 in combinations(top, 2):
            if g1 in piv.columns and g2 in piv.columns:
                if (piv[g1] & piv[g2]).sum() >= self.min_comut_patients:
                    pairs.append(tuple(sorted([g1, g2])))
        self.comut_pairs_ = sorted(set(pairs))

    # -----------------------------------------------------------------
    # CYTOGENETICS TOKENIZER — rule-based parsing, auto-discovered tokens
    # -----------------------------------------------------------------
    @staticmethod
    def _tokenize_cyto(s):
        """Parse ISCN cytogenetics string into a set of abnormality tokens."""
        tokens = set()
        s = s.lower().strip()
        # Deletions
        for m in re.finditer(r"del\((\w+)\)", s):
            tokens.add(f"del_{m.group(1)}")
        # Translocations (sort chromosomes for consistency)
        for m in re.finditer(r"t\(([^)]+)\)", s):
            chroms = sorted(re.findall(r"\w+", m.group(1)))
            if chroms:
                tokens.add("t_" + "_".join(chroms))
        # Inversions
        for m in re.finditer(r"inv\((\w+)\)", s):
            tokens.add(f"inv_{m.group(1)}")
        # Monosomies
        for m in re.finditer(r"(?:^|[,/\s])-(\d+)", s):
            tokens.add(f"mono_{m.group(1)}")
        # Trisomies
        for m in re.finditer(r"\+(\d+)", s):
            tokens.add(f"tri_{m.group(1)}")
        # Additions
        for m in re.finditer(r"add\((\w+)\)", s):
            tokens.add(f"add_{m.group(1)}")
        # Derivatives
        for m in re.finditer(r"der\((\w+)\)", s):
            tokens.add(f"der_{m.group(1)}")
        # Duplications
        for m in re.finditer(r"dup\((\w+)\)", s):
            tokens.add(f"dup_{m.group(1)}")
        # Isochromosomes
        for m in re.finditer(r"(?:i|idic)\((\w+)\)", s):
            tokens.add(f"iso_{m.group(1)}")
        # Sex chromosome loss
        if "-y" in s:
            tokens.add("loss_y")
        if "-x" in s:
            tokens.add("loss_x")
        return tokens

    # -----------------------------------------------------------------
    # TRANSFORM — build feature matrix
    # -----------------------------------------------------------------
    def transform(self, clinical_df, molecular_df):
        assert self._fitted, "Call fit() first."
        ids = clinical_df["ID"].values
        feat = pd.DataFrame({"ID": ids})

        feat = self._add_clinical(clinical_df, feat)
        feat = self._add_cyto(clinical_df, feat)
        feat = self._add_molecular(molecular_df, feat)
        # feat = self._add_comut(molecular_df, feat)  # disabled: too many zero-imp features
        feat = self._add_chr_counts(molecular_df, feat)
        feat = self._add_targeted_interactions(clinical_df, molecular_df, feat)
        feat = self._add_chromosome_diversity(molecular_df, feat)

        feat = feat.replace([np.inf, -np.inf], np.nan)
        return feat

    # --- clinical ---
    def _add_clinical(self, cdf, feat):
        # Raw values
        for c in self.CLIN_COLS:
            feat[c] = cdf[c].values.astype(float)

        # Log transforms
        for c in self.CLIN_COLS:
            vals = cdf[c].values.astype(float)
            feat[f"{c}_log"] = np.log1p(np.clip(vals, 0, None))

        # All pairwise ratios and products
        for c1, c2 in combinations(self.CLIN_COLS, 2):
            v1 = cdf[c1].values.astype(float)
            v2 = cdf[c2].values.astype(float)
            feat[f"ratio_{c1}_{c2}"] = v1 / (v2 + 0.01)
            feat[f"prod_{c1}_{c2}"] = v1 * v2

        return feat

    # --- cytogenetics ---
    def _add_cyto(self, cdf, feat):
        cyto = cdf["CYTOGENETICS"]
        feat["cyto_missing"] = cyto.isna().astype(int).values

        n_clones_list, n_abn_list, normal_list, chr_count_list = [], [], [], []
        for s in cyto:
            if pd.isna(s):
                n_clones_list.append(0)
                n_abn_list.append(0)
                normal_list.append(0)
                chr_count_list.append(0)
                continue
            s_low = str(s).lower().strip()

            m_chr = re.match(r"^(\d+),", s_low)
            chr_count = int(m_chr.group(1)) if m_chr else 0
            if chr_count > 90:
                chr_count = 0
            chr_count_list.append(chr_count)

            clones = s_low.split("/")
            n_clones_list.append(len(clones))

            n_struct = len(re.findall(r"(del|inv|t|dup|add|ins|der)\(", s_low))
            n_mono = len(re.findall(r"(?:^|[,/\s])-(\d+)", s_low))
            n_tri = len(re.findall(r"\+(\d+)", s_low))
            n_abn_list.append(n_struct + n_mono + n_tri)

            first = clones[0].strip()
            normal_list.append(int(bool(re.match(r"^46,(xx|xy)(\[\d+\])?$", first))))

        feat["cyto_n_clones"] = n_clones_list
        feat["cyto_n_abn"] = n_abn_list
        feat["cyto_normal"] = normal_list
        feat["cyto_complex"] = (np.array(n_abn_list) >= 3).astype(int)
        feat["cyto_chr_count"] = chr_count_list

        # Explicit cyto binary flags removed — covered by auto-discovered tokens

        # Auto-discovered token indicators
        for tok in self.cyto_tokens_:
            vals = []
            for s in cyto:
                if pd.isna(s):
                    vals.append(0)
                else:
                    vals.append(int(tok in self._tokenize_cyto(str(s))))
            feat[f"cyto_{tok}"] = vals

        return feat

    # --- molecular ---
    def _add_molecular(self, mdf, feat):
        ids = feat["ID"].values
        id_set = set(ids)
        mol = (
            mdf[mdf["ID"].isin(id_set)].copy()
            if mdf is not None and not mdf.empty
            else pd.DataFrame()
        )

        # Default zeros for empty molecular data
        if mol.empty:
            for c in [
                "n_mutations",
                "n_unique_genes",
                "mean_vaf",
                "max_vaf",
                "std_vaf",
                "min_vaf",
                "n_clonal",
                "n_subclonal",
                "clonal_ratio",
                "n_distinct_effects",
            ]:
                feat[c] = 0
            for g in self.genes_:
                feat[f"has_{g}"] = 0
                feat[f"vaf_{g}"] = 0.0
            for e in self.effects_:
                feat[f"eff_{e}"] = 0
            return feat

        grp = mol.groupby("ID")

        # --- Aggregate statistics ---
        agg_dict = {
            "n_mutations": ("GENE", "size"),
            "n_unique_genes": ("GENE", "nunique"),
        }
        agg = grp.agg(**agg_dict).reset_index()

        if "VAF" in mol.columns:
            vagg = (
                grp["VAF"]
                .agg(mean_vaf="mean", max_vaf="max", std_vaf="std", min_vaf="min")
                .reset_index()
            )
            agg = agg.merge(vagg, on="ID")

        if "EFFECT" in mol.columns:
            ne = grp["EFFECT"].nunique().reset_index(name="n_distinct_effects")
            agg = agg.merge(ne, on="ID")

        feat = feat.merge(agg, on="ID", how="left")
        for c in agg.columns.drop("ID"):
            feat[c] = feat[c].fillna(0)

        # Clonal vs subclonal counts
        if "VAF" in mol.columns:
            for thresh, label in [
                (0.2, "vaf_gt_20"),
                (0.3, "vaf_gt_30"),
                (0.5, "vaf_gt_50"),
            ]:
                nc = (
                    mol[mol["VAF"] > thresh]
                    .groupby("ID")
                    .size()
                    .reset_index(name=f"n_{label}")
                )
                feat = feat.merge(nc, on="ID", how="left")
                feat[f"n_{label}"] = feat[f"n_{label}"].fillna(0)
            feat["clonal_ratio"] = feat["n_vaf_gt_30"] / (feat["n_mutations"] + 1)
            feat["high_clone_ratio"] = feat["n_vaf_gt_50"] / (feat["n_mutations"] + 1)
        else:
            for label in ["vaf_gt_20", "vaf_gt_30", "vaf_gt_50"]:
                feat[f"n_{label}"] = 0
            feat["clonal_ratio"] = 0
            feat["high_clone_ratio"] = 0

        # --- Per-gene features via efficient pivot ---
        gm = mol[mol["GENE"].isin(self.genes_)]
        if not gm.empty:
            # Binary: has mutation in gene
            has_piv = gm.groupby(["ID", "GENE"]).size().unstack(fill_value=0)
            cnt_piv = has_piv.copy()
            has_piv = (has_piv > 0).astype(int)

            # Max VAF per gene
            has_vaf = "VAF" in mol.columns
            if has_vaf:
                vaf_piv = gm.groupby(["ID", "GENE"])["VAF"].max().unstack(fill_value=0)

            # Ensure all vocabulary genes are present
            for g in self.genes_:
                if g not in has_piv.columns:
                    has_piv[g] = 0
                    cnt_piv[g] = 0
                if has_vaf and g not in vaf_piv.columns:
                    vaf_piv[g] = 0.0

            has_piv = has_piv[self.genes_]
            cnt_piv = cnt_piv[self.genes_]
            has_piv.columns = [f"has_{g}" for g in self.genes_]
            # cnt_piv columns not added — mostly zero-imp in LGBM

            feat = feat.merge(has_piv, left_on="ID", right_index=True, how="left")
            # feat = feat.merge(cnt_piv, left_on="ID", right_index=True, how="left")

            if has_vaf:
                vaf_piv = vaf_piv[self.genes_]
                vaf_piv.columns = [f"vaf_{g}" for g in self.genes_]
                feat = feat.merge(vaf_piv, left_on="ID", right_index=True, how="left")
            else:
                for g in self.genes_:
                    feat[f"vaf_{g}"] = 0.0
        else:
            for g in self.genes_:
                feat[f"has_{g}"] = 0
                feat[f"vaf_{g}"] = 0.0
                feat[f"cnt_{g}"] = 0

        # Fill NaNs for gene features
        for g in self.genes_:
            feat[f"has_{g}"] = feat[f"has_{g}"].fillna(0).astype(int)
            feat[f"vaf_{g}"] = feat[f"vaf_{g}"].fillna(0.0)

        # --- Per-effect counts ---
        if "EFFECT" in mol.columns and self.effects_:
            em = mol[mol["EFFECT"].isin(self.effects_)]
            if not em.empty:
                ep = em.groupby(["ID", "EFFECT"]).size().unstack(fill_value=0)
                for e in self.effects_:
                    if e not in ep.columns:
                        ep[e] = 0
                ep = ep[self.effects_]
                ep.columns = [f"eff_{e}" for e in self.effects_]
                feat = feat.merge(ep, left_on="ID", right_index=True, how="left")

        for e in self.effects_:
            col = f"eff_{e}"
            if col not in feat.columns:
                feat[col] = 0
            feat[col] = feat[col].fillna(0).astype(int)

        return feat

    # --- co-mutation indicators ---
    def _add_comut(self, mdf, feat):
        if not self.comut_pairs_:
            return feat
        ids = feat["ID"].values
        id_set = set(ids)
        mol = (
            mdf[mdf["ID"].isin(id_set)]
            if mdf is not None and not mdf.empty
            else pd.DataFrame()
        )
        if mol.empty:
            for g1, g2 in self.comut_pairs_:
                feat[f"comut_{g1}_{g2}"] = 0
            return feat

        patient_genes = mol.groupby("ID")["GENE"].apply(set).to_dict()
        for g1, g2 in self.comut_pairs_:
            feat[f"comut_{g1}_{g2}"] = [
                int(
                    g1 in patient_genes.get(pid, set())
                    and g2 in patient_genes.get(pid, set())
                )
                for pid in ids
            ]
        return feat

    # --- per-chromosome mutation counts ---
    def _add_chr_counts(self, mdf, feat):
        if not self.chroms_:
            return feat
        ids = feat["ID"].values
        id_set = set(ids)
        mol = (
            mdf[mdf["ID"].isin(id_set)].copy()
            if mdf is not None and not mdf.empty
            else pd.DataFrame()
        )
        if mol.empty or "CHR" not in mol.columns:
            for ch in self.chroms_:
                feat[f"chr_{ch}_muts"] = 0
            return feat

        mol["_chr"] = mol["CHR"].astype(str).str.strip()
        for ch in self.chroms_:
            cm = (
                mol[mol["_chr"] == ch]
                .groupby("ID")
                .size()
                .reset_index(name=f"chr_{ch}_muts")
            )
            feat = feat.merge(cm, on="ID", how="left")
            feat[f"chr_{ch}_muts"] = feat[f"chr_{ch}_muts"].fillna(0).astype(int)
        return feat

    # --- distinct chromosomes affected ---
    def _add_chromosome_diversity(self, mdf, feat):
        ids = feat["ID"].values
        id_set = set(ids)
        mol = (
            mdf[mdf["ID"].isin(id_set)].copy()
            if mdf is not None and not mdf.empty
            else pd.DataFrame()
        )
        if mol.empty or "CHR" not in mol.columns:
            feat["n_chr_affected"] = 0
            return feat
        mol["_chr"] = mol["CHR"].astype(str).str.strip()
        chr_counts = (
            mol.groupby("ID")["_chr"].nunique().reset_index(name="n_chr_affected")
        )
        feat = feat.merge(chr_counts, on="ID", how="left")
        feat["n_chr_affected"] = feat["n_chr_affected"].fillna(0).astype(int)
        return feat

    # --- targeted high-value interaction features ---
    def _add_targeted_interactions(self, cdf, mdf, feat):
        ids = feat["ID"].values
        id_set = set(ids)
        mol = (
            mdf[mdf["ID"].isin(id_set)].copy()
            if mdf is not None and not mdf.empty
            else pd.DataFrame()
        )

        if mol.empty:
            feat["npm1_no_flt3itd"] = 0
            feat["tp53_multihit"] = 0
            feat["flt3_itd"] = 0
            feat["flt3_itd_vaf"] = 0.0
            feat["monosomal_karyotype"] = 0
            feat["n_monosomies"] = 0
            feat["path_epigenetic_count"] = 0
            feat["path_splicing_count"] = 0
            feat["path_signaling_count"] = 0
            feat["cyto_major_clone_frac"] = 0.0
            feat["max_driver_vaf"] = 0.0
            feat["n_driver_mutations"] = 0
            return feat

        patient_genes = mol.groupby("ID")["GENE"].apply(set).to_dict()

        itd_patients = set()
        if "EFFECT" in mol.columns:
            itd_rows = mol[
                (mol["GENE"] == "FLT3")
                & (mol["EFFECT"].astype(str).str.lower().str.contains("itd"))
            ]
            itd_patients = set(itd_rows["ID"].values)
        if "PROTEIN_CHANGE" in mol.columns:
            itd_rows2 = mol[
                (mol["GENE"] == "FLT3")
                & (mol["PROTEIN_CHANGE"].astype(str).str.upper().str.contains("ITD"))
            ]
            itd_patients |= set(itd_rows2["ID"].values)

        tp53_counts = mol[mol["GENE"] == "TP53"].groupby("ID").size()
        tp53_multi_pids = set(tp53_counts[tp53_counts >= 2].index)

        flt3_itd_vaf_map = {}
        if "VAF" in mol.columns:
            itd_mol = mol[(mol["GENE"] == "FLT3") & mol["ID"].isin(itd_patients)]
            if not itd_mol.empty:
                flt3_itd_vaf_map = itd_mol.groupby("ID")["VAF"].max().to_dict()

        max_vaf_map = {}
        n_driver_map = {}
        if "VAF" in mol.columns:
            driver_genes = {
                "FLT3",
                "NPM1",
                "TP53",
                "DNMT3A",
                "IDH1",
                "IDH2",
                "RUNX1",
                "CEBPA",
                "WT1",
                "ASXL1",
            }
            driver_mol = mol[mol["GENE"].isin(driver_genes)]
            if not driver_mol.empty:
                max_vaf_map = driver_mol.groupby("ID")["VAF"].max().to_dict()
                n_driver_map = driver_mol.groupby("ID")["GENE"].nunique().to_dict()

        cyto_map = dict(zip(cdf["ID"].values, cdf["CYTOGENETICS"].values))

        epigenetic_genes = {"TET2", "DNMT3A", "IDH1", "IDH2", "ASXL1", "EZH2", "BCOR"}
        splicing_genes = {"SF3B1", "SRSF2", "U2AF1", "ZRSR2"}
        signaling_genes = {
            "FLT3",
            "NRAS",
            "KRAS",
            "KIT",
            "PTPN11",
            "JAK2",
            "CBL",
            "MPL",
        }

        npm1_no_flt3 = []
        tp53_multihit = []
        flt3_itd = []
        flt3_itd_vaf = []
        monosomal_karyotype = []
        n_monosomies = []
        path_epi = []
        path_spl = []
        path_sig = []
        clone_frac = []
        max_driver_vaf = []
        n_driver_mut = []

        for pid in ids:
            genes = patient_genes.get(pid, set())

            npm1_no_flt3.append(int("NPM1" in genes and pid not in itd_patients))
            tp53_multihit.append(int(pid in tp53_multi_pids))
            flt3_itd.append(int(pid in itd_patients))
            flt3_itd_vaf.append(flt3_itd_vaf_map.get(pid, 0.0))

            cyto_str = cyto_map.get(pid, "")
            if pd.isna(cyto_str):
                cyto_str = ""
            s_low = cyto_str.lower().strip()

            monos = re.findall(r"(?:^|[,/\s])-(\d+)", s_low)
            n_mon = len(monos)
            n_monosomies.append(n_mon)

            n_struct = len(re.findall(r"(del|inv|t|dup|add|ins|der)\(", s_low))
            n_tri = len(re.findall(r"\+(\d+)", s_low))
            mk = 1 if n_mon >= 2 or (n_mon >= 1 and n_struct > 0) else 0
            monosomal_karyotype.append(mk)

            path_epi.append(len(genes & epigenetic_genes))
            path_spl.append(len(genes & splicing_genes))
            path_sig.append(len(genes & signaling_genes))

            clones = s_low.split("/")
            cell_counts = []
            for clone in clones:
                m = re.search(r"\[(\d+)\]", clone)
                if m:
                    cell_counts.append(int(m.group(1)))
            total = sum(cell_counts)
            clone_frac.append(max(cell_counts) / total if total > 0 else 0.0)

            max_driver_vaf.append(max_vaf_map.get(pid, 0.0))
            n_driver_mut.append(n_driver_map.get(pid, 0))

        feat["npm1_no_flt3itd"] = npm1_no_flt3
        feat["tp53_multihit"] = tp53_multihit
        feat["flt3_itd"] = flt3_itd
        feat["flt3_itd_vaf"] = flt3_itd_vaf
        feat["monosomal_karyotype"] = monosomal_karyotype
        feat["n_monosomies"] = n_monosomies
        feat["path_epigenetic_count"] = path_epi
        feat["path_splicing_count"] = path_spl
        feat["path_signaling_count"] = path_sig
        feat["cyto_major_clone_frac"] = clone_frac
        feat["max_driver_vaf"] = max_driver_vaf
        feat["n_driver_mutations"] = n_driver_mut

        clinical_cols_for_interact = ["BM_BLAST", "WBC", "ANC", "HB", "PLT"]
        for cc in clinical_cols_for_interact:
            cvals = cdf.set_index("ID").reindex(ids)[cc].values.astype(float)
            for flag, flag_vals in [
                ("flt3_itd", flt3_itd),
                (
                    "tp53_mut",
                    [int("TP53" in patient_genes.get(pid, set())) for pid in ids],
                ),
                (
                    "npm1_mut",
                    [int("NPM1" in patient_genes.get(pid, set())) for pid in ids],
                ),
            ]:
                fv = np.array(flag_vals, dtype=float)
                feat[f"{cc}_x_{flag}"] = cvals * fv
                feat[f"{cc}_x_{flag}_ratio"] = cvals / (fv + 0.01)

        return feat


# =====================================================================
# CONVENIENCE FUNCTION
# =====================================================================
def build_features(data, builder_params=None):
    """
    Build feature matrices from raw data.

    Args:
        data: dict from load_raw_data() with keys:
              clinical_train, clinical_test, molecular_train, molecular_test, target
        builder_params: optional dict of AutoFeatureBuilder kwargs

    Returns:
        train_feat: DataFrame (ID + feature columns)
        test_feat:  DataFrame (ID + feature columns)
        builder:    fitted AutoFeatureBuilder instance
    """
    params = builder_params or {}
    builder = AutoFeatureBuilder(**params)
    builder.fit(data["clinical_train"], data["molecular_train"])

    train_feat = builder.transform(data["clinical_train"], data["molecular_train"])
    test_feat = builder.transform(data["clinical_test"], data["molecular_test"])

    return train_feat, test_feat, builder
