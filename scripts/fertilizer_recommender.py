from typing import Dict, List


def get_fertilizer_recommendation(N: float, P: float, K: float, pH: float, predicted_crop: str) -> Dict[str, List[str]]:
    """
    Provide simple rule-based soil and fertilizer recommendations.

    Returns a dictionary with keys:
      - soil_status: "Good" or "Needs Improvement"
      - recommendations: list of short actionable suggestions
    """
    try:
        N = float(N)
        P = float(P)
        K = float(K)
        pH = float(pH)
    except Exception:
        return {
            "soil_status": "Needs Improvement",
            "recommendations": [
                "Verify N, P, K, and pH inputs are valid numbers.",
            ],
        }

    recs: List[str] = []

    # Generic target bands (can be tuned per dataset/region)
    target = {
        "N": (40.0, 100.0),
        "P": (30.0, 90.0),
        "K": (30.0, 90.0),
        "pH": (6.0, 7.5),
    }

    lowN, highN = target["N"]
    lowP, highP = target["P"]
    lowK, highK = target["K"]
    lowPH, highPH = target["pH"]

    # pH first (critical for nutrient availability)
    if pH < lowPH:
        recs.append("pH is low: apply agricultural lime or dolomite to raise pH towards 6.5–7.0.")
    elif pH > highPH:
        recs.append("pH is high: apply elemental sulfur/acidifying fertilizers and add organic matter.")
    else:
        recs.append("pH is in an optimal range for most crops.")

    # Nitrogen
    if N < lowN:
        recs.append("Nitrogen is low: apply urea or ammonium sulfate in split doses.")
    elif N > highN:
        recs.append("Nitrogen is high: reduce N applications and avoid excessive urea.")
    else:
        recs.append("Nitrogen level looks adequate.")

    # Phosphorus
    if P < lowP:
        recs.append("Phosphorus is low: apply DAP/SSP at land prep and incorporate well.")
    elif P > highP:
        recs.append("Phosphorus is high: skip P fertilizer this cycle; consider soil test later.")
    else:
        recs.append("Phosphorus level looks adequate.")

    # Potassium
    if K < lowK:
        recs.append("Potassium is low: apply MOP (potash); split for sandy soils.")
    elif K > highK:
        recs.append("Potassium is high: avoid K fertilizer this cycle; monitor leaf scorch risk.")
    else:
        recs.append("Potassium level looks adequate.")

    # Crop-specific quick notes
    crop = (predicted_crop or "").strip().lower()
    if crop:
        if crop in {"rice", "paddy"}:
            recs.append("For rice: prefer ammonium forms of N; maintain field moisture; split N at tillering.")
        elif crop == "wheat":
            recs.append("For wheat: neutral pH is ideal; ensure basal P and early N top-dress.")
        elif crop in {"maize", "corn"}:
            recs.append("For maize: early N is critical; ensure adequate K for stalk strength.")
        elif crop == "banana":
            recs.append("For banana: high K demand; mulch and regular fertigation improve bunch size.")
        elif crop == "jute":
            recs.append("For jute: balanced N and P; avoid waterlogging during early growth.")
        elif crop == "cotton":
            recs.append("For cotton: ensure K and micronutrients (B/Zn) for boll retention.")

    # Soil status
    needs = any([
        pH < lowPH or pH > highPH,
        N < lowN or N > highN,
        P < lowP or P > highP,
        K < lowK or K > highK,
    ])

    status = "Needs Improvement" if needs else "Good"

    return {
        "soil_status": status,
        "recommendations": recs,
    }


