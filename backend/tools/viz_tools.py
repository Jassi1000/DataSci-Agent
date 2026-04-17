# backend/tools/viz_tools.py
import base64
from io import BytesIO


def figure_to_base64(plt) -> str:
    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

