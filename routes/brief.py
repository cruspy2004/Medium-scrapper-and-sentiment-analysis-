from flask import Blueprint, request, jsonify

from analysis.brief import generate_brief

brief_bp = Blueprint('brief', __name__)


@brief_bp.route('/brief', methods=['POST'])
def create_brief():
    """
    JSON endpoint for on-demand content brief generation.
    Called client-side via fetch() when user clicks a gap card.
    No page reload — returns JSON directly.
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    brief = generate_brief(data)
    return jsonify(brief)
