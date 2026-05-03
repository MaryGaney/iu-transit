"""
app/api/travel_agent.py
────────────────────────
LLM travel agent using HuggingFace Inference API (Mistral-7B-Instruct).
Falls back to rule-based schedule lookup if HF is unavailable.

POST /api/travel-agent/chat
GET  /api/travel-agent/status
"""

import os
import json
import httpx
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.core.database import get_db
from app.core.logging import logger
from app.services.weather import get_current_weather
from app.services.gtfs_realtime import get_current_positions

router = APIRouter(prefix="/api/travel-agent", tags=["travel_agent"])

HF_MODEL   = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


class ChatResponse(BaseModel):
    model_config = {'protected_namespaces': ()}
    reply: str
    model_used: str
    context_used: dict


# ── Building locations — comprehensive IU campus index ────────────────────────
# Keys are ALL the ways a student might refer to a building.
# The more aliases the better — the matcher is substring-based.

BUILDING_LOCATIONS: dict[str, tuple[float, float]] = {
    # IMU / Memorial Union (many aliases)
    "imu":               (39.16790, -86.52440),
    "indiana memorial union": (39.16790, -86.52440),
    "memorial union":    (39.16790, -86.52440),
    "union":             (39.16790, -86.52440),
    "imunion":           (39.16790, -86.52440),

    # Academic buildings
    "ballantine":        (39.16860, -86.52250),
    "luddy":             (39.17330, -86.52020),
    "informatics":       (39.17330, -86.52020),
    "info school":       (39.17330, -86.52020),
    "spea":              (39.17230, -86.51390),
    "public affairs":    (39.17230, -86.51390),
    "environmental affairs": (39.17230, -86.51390),
    "jordan":            (39.17180, -86.52470),
    "jordan hall":       (39.17180, -86.52470),
    "chemistry":         (39.17000, -86.52050),
    "psychology":        (39.17060, -86.51960),
    "psych":             (39.17060, -86.51960),
    "business":          (39.16950, -86.51500),
    "hodge":             (39.16950, -86.51500),
    "kelley":            (39.16950, -86.51500),
    "law school":        (39.16940, -86.52200),
    "law":               (39.16940, -86.52200),
    "maurer":            (39.16940, -86.52200),
    "music":             (39.17520, -86.51730),
    "musical arts":      (39.17520, -86.51730),
    "mac":               (39.17520, -86.51730),
    "wylie":             (39.16780, -86.52200),
    "woodburn":          (39.16780, -86.52370),
    "franklin":          (39.16650, -86.52250),
    "swain":             (39.17200, -86.51810),
    "wells":             (39.17090, -86.52280),
    "wells library":     (39.17090, -86.52280),
    "library":           (39.17090, -86.52280),
    "rawles":            (39.16690, -86.52220),
    "lindley":           (39.16620, -86.52340),
    "eigenmann":         (39.16320, -86.51170),
    "teter":             (39.16600, -86.51800),
    "fine arts":         (39.17510, -86.51660),
    "kirkwood":          (39.16800, -86.52100),
    "atwater":           (39.16560, -86.51960),
    "sycamore":          (39.16870, -86.52190),
    "morrison":          (39.16780, -86.52440),
    "global":            (39.17040, -86.51480),
    "global and international": (39.17040, -86.51480),
    "geology":           (39.17080, -86.52320),
    "simon":             (39.17080, -86.52320),

    # Common landmarks / areas
    "sample gates":      (39.16880, -86.52360),
    "sample gate":       (39.16880, -86.52360),
    "third street":      (39.16500, -86.52500),
    "10th street":       (39.17000, -86.52500),
    "stadium":           (39.18320, -86.52440),
    "memorial stadium":  (39.18320, -86.52440),
    "assembly hall":     (39.18400, -86.52570),
    "simon skjodt":      (39.18400, -86.52570),
    "rec sports":        (39.17600, -86.52200),
    "srsc":              (39.17600, -86.52200),

    # Dorms / quads
    "forest":            (39.17050, -86.50820),
    "forest quad":       (39.17050, -86.50820),
    "read":              (39.16890, -86.52150),
    "read hall":         (39.16890, -86.52150),
    "wright":            (39.17200, -86.51300),
    "education":         (39.17200, -86.51300),

    # Off-campus
    "college mall":      (39.15800, -86.50500),
    "target":            (39.15800, -86.50500),
    "walmart":           (39.18500, -86.56500),
    "ivy tech":          (39.15500, -86.53000),
    "bloomington hospital": (39.17200, -86.49800),
    "iu health":         (39.17200, -86.49800),
}


def _extract_buildings(query: str) -> dict[str, tuple]:
    """
    Find all building references in a query string.
    Case-insensitive substring match against the full alias table.
    Returns { alias_found: (lat, lng) }
    """
    q = query.lower()
    found = {}
    # Sort by length descending so longer matches win over shorter ones
    for name, coords in sorted(BUILDING_LOCATIONS.items(), key=lambda x: -len(x[0])):
        if name in q and name not in found:
            found[name] = coords
    return found


async def _get_transit_context(db: AsyncSession, query: str) -> dict:
    """Build live context for the LLM prompt."""
    now = datetime.now()

    context = {
        "current_time":    now.strftime("%I:%M %p"),
        "current_time_24": now.strftime("%H:%M:%S"),
        "day_of_week":     now.strftime("%A"),
        "date":            now.strftime("%B %d, %Y"),
    }

    # Weather
    weather = get_current_weather()
    if weather:
        context["weather"] = {
            "temp_f":     round(weather.get("temperature_c", 20) * 9/5 + 32),
            "conditions": ("Rain"  if weather.get("is_raining") else
                          "Snow"  if weather.get("is_snowing") else "Clear"),
            "severity":   weather.get("weather_severity", 0),
        }

    # Live buses
    positions = await get_current_positions()
    context["live_buses"] = [
        {
            "vehicle_id": p["vehicle_id"],
            "route_id":   p["route_id"],
            "lat":        round(p["lat"], 4),
            "lng":        round(p["lng"], 4),
        }
        for p in (positions or [])[:20]
    ]

    # All routes
    routes_result = await db.execute(text(
        "SELECT route_id, route_short_name, route_long_name FROM routes ORDER BY route_short_name"
    ))
    context["routes"] = [
        {"id": r[0], "short": r[1], "name": r[2]}
        for r in routes_result.fetchall()
    ]

    # Buildings detected in query
    buildings_found = _extract_buildings(query)
    context["buildings_detected"] = list(buildings_found.keys())

    # Find nearby stops for each detected building
    nearby_stops = []
    for building_name, (blat, blng) in buildings_found.items():
        stops_result = await db.execute(text("""
            SELECT stop_id, stop_name, stop_lat, stop_lon,
                   (ABS(stop_lat - :lat) + ABS(stop_lon - :lng)) as dist
            FROM stops
            ORDER BY dist ASC
            LIMIT 3
        """), {"lat": blat, "lng": blng})
        for row in stops_result.fetchall():
            nearby_stops.append({
                "near":      building_name,
                "stop_id":   row[0],
                "stop_name": row[1],
            })
    context["stops_near_mentioned_places"] = nearby_stops

    # Upcoming arrivals at those stops
    if nearby_stops:
        stop_ids = list({s["stop_id"] for s in nearby_stops})
        t_now = now.strftime("%H:%M:%S")
        t_end = (now + timedelta(minutes=90)).strftime("%H:%M:%S")

        if stop_ids:
            placeholders = ",".join(f"'{sid}'" for sid in stop_ids)
            arrivals_result = await db.execute(text(f"""
                SELECT st.stop_id, st.arrival_time,
                       t.route_id, r.route_short_name, r.route_long_name,
                       t.trip_headsign
                FROM stop_times st
                JOIN trips t ON st.trip_id = t.trip_id
                JOIN routes r ON t.route_id = r.route_id
                WHERE st.stop_id IN ({placeholders})
                  AND st.arrival_time >= :t_now
                  AND st.arrival_time <= :t_end
                ORDER BY st.stop_id, st.arrival_time
                LIMIT 40
            """), {"t_now": t_now, "t_end": t_end})

            context["upcoming_arrivals"] = [
                {
                    "stop_id":    r[0],
                    "arrival":    r[1],
                    "route":      r[2],
                    "route_short": r[3],
                    "route_name":  r[4],
                    "headsign":    r[5],
                }
                for r in arrivals_result.fetchall()
            ]
    else:
        context["upcoming_arrivals"] = []

    # Classes releasing soon (crowd signal)
    t_start = now.strftime("%H:%M:%S")
    t_end20 = (now + timedelta(minutes=20)).strftime("%H:%M:%S")
    releasing = await db.execute(text("""
        SELECT cs.building_code, cs.end_time, SUM(cs.enrollment) as total
        FROM class_sections cs
        WHERE cs.is_in_person=1
          AND cs.end_time >= :t_start AND cs.end_time <= :t_end
          AND cs.enrollment > 0
        GROUP BY cs.building_code, cs.end_time
        ORDER BY total DESC LIMIT 6
    """), {"t_start": t_start, "t_end": t_end20})
    rows = releasing.fetchall()
    if rows:
        context["classes_releasing_soon"] = [
            {"building": r[0], "at": str(r[1]), "students": r[2]}
            for r in rows
        ]

    return context


def _build_system_prompt(context: dict) -> str:
    return f"""You are a helpful IU Bloomington transit assistant. Give students specific, actionable bus advice.

CURRENT CONDITIONS:
- Time: {context.get('current_time')} on {context.get('day_of_week')}, {context.get('date')}
- Weather: {json.dumps(context.get('weather', {}))}
- Live buses on road: {len(context.get('live_buses', []))}

ROUTES AVAILABLE:
{json.dumps(context.get('routes', []), indent=2)}

STOPS NEAR PLACES MENTIONED:
{json.dumps(context.get('stops_near_mentioned_places', []), indent=2)}

UPCOMING ARRIVALS (next 90 min):
{json.dumps(context.get('upcoming_arrivals', []), indent=2)}

CROWD ALERT — Classes releasing next 20 min:
{json.dumps(context.get('classes_releasing_soon', []), indent=2)}

INSTRUCTIONS:
1. Give a SPECIFIC answer: which route number, which stop name, what time to leave.
2. Add 5 min walk time for nearby buildings, 10+ min for distant ones.
3. If classes are releasing nearby, warn the bus may be full (+2-4 min delay).
4. Bad weather (rain/snow/extreme temp) = leave 3-5 min early.
5. Give a confidence estimate ("~80% chance you make it").
6. Format: Route, Stop, Leave By, Notes. Keep it under 120 words.
7. If no arrival data found, say so honestly and suggest etaspot.net."""


def _rule_based_response(context: dict, query: str) -> str:
    """Structured fallback when HF is unavailable."""
    lines = []
    arrivals = context.get("upcoming_arrivals", [])
    buildings = context.get("buildings_detected", [])

    if not buildings:
        lines.append("I couldn't identify specific campus buildings in your question.")
        lines.append("Try mentioning building names like 'Ballantine', 'Luddy Hall', 'IMU', or 'law school'.")
    elif not arrivals:
        lines.append(f"I found stops near: {', '.join(buildings)}")
        lines.append(f"No scheduled arrivals in the next 90 minutes (checked at {context.get('current_time')}).")
        lines.append("Either buses aren't running right now, or the stops table needs to be loaded.")
        lines.append("Check live info at etaspot.net")
    else:
        lines.append(f"🚌 Upcoming buses near {', '.join(buildings)}:")
        for a in arrivals[:5]:
            lines.append(f"  Route {a['route_short']} ({a['route_name']}) → {a['headsign'] or 'terminus'} at {a['arrival']}")
        crowd = context.get("classes_releasing_soon", [])
        if crowd:
            total = sum(c["students"] for c in crowd)
            lines.append(f"⚠️ ~{total} students releasing nearby — expect crowded buses.")
        w = context.get("weather", {})
        if w.get("severity", 0) > 0.3:
            lines.append(f"🌧 {w.get('conditions')} — leave a few minutes early.")

    return "\n".join(lines)


async def _call_huggingface(prompt_prefix: str, messages: list[dict]) -> tuple[str, str]:
    """Call HuggingFace Inference API with Mistral-7B-Instruct."""
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        return "", ""

    full_messages = [{"role": "system", "content": prompt_prefix}] + messages
    chat_url = f"https://api-inference.huggingface.co/models/{HF_MODEL}/v1/chat/completions"

    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                chat_url,
                headers={
                    "Authorization": f"Bearer {hf_token}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model":       HF_MODEL,
                    "messages":    full_messages,
                    "max_tokens":  450,
                    "temperature": 0.3,
                    "stream":      False,
                },
            )

            if resp.status_code == 200:
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                return text.strip(), f"Mistral-7B via HuggingFace"

            elif resp.status_code == 503:
                return (
                    "⏳ The AI model is warming up (first request takes ~20s on free tier). "
                    "Please ask again in a moment.",
                    "hf/loading"
                )
            elif resp.status_code == 429:
                return (
                    "⚠️ HuggingFace free tier rate limit reached. "
                    "Try again in a few minutes.",
                    ""
                )
            elif resp.status_code == 401:
                logger.error("HF 401 — token invalid or missing inference scope")
                return "", ""
            else:
                logger.warning(f"HF API {resp.status_code}: {resp.text[:200]}")
                return "", ""

    except httpx.TimeoutException:
        return (
            "⏳ AI response timed out (model cold start). Please try again in 20 seconds.",
            ""
        )
    except Exception as e:
        logger.error(f"HF API error: {e}")
        return "", ""


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    context = await _get_transit_context(db, request.message)
    prompt  = _build_system_prompt(context)
    messages = list(request.history) + [{"role": "user", "content": request.message}]

    reply, model_used = await _call_huggingface(prompt, messages)

    if not reply or reply.startswith("hf/"):
        reply      = _rule_based_response(context, request.message)
        model_used = "rule-based"

    return ChatResponse(
        reply=reply,
        model_used=model_used,
        context_used={
            "routes_loaded":    len(context.get("routes", [])),
            "live_buses":       len(context.get("live_buses", [])),
            "arrivals_found":   len(context.get("upcoming_arrivals", [])),
            "buildings_detected": context.get("buildings_detected", []),
            "weather":          context.get("weather"),
        },
    )


@router.get("/status")
async def llm_status():
    """Check HuggingFace configuration status."""
    hf_token    = os.environ.get("HF_TOKEN", "")
    has_token   = bool(hf_token)
    token_valid = False

    if has_token:
        # Use the models API to validate — more reliable than /api/whoami
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                probe = await client.get(
                    f"https://api-inference.huggingface.co/models/{HF_MODEL}",
                    headers={"Authorization": f"Bearer {hf_token}"},
                )
                token_valid = probe.status_code in (200, 503)
                # 503 = model loading = token valid, model just cold
        except Exception:
            pass

    return {
        "active_backend": "huggingface" if token_valid else "rule-based",
        "huggingface": {
            "configured":   has_token,
            "token_valid":  token_valid,
            "model":        HF_MODEL,
        },
        "fallback": "rule-based (always available)",
        "setup_instructions": (
            "Add HF_TOKEN to Railway environment variables. "
            "Get a free token at huggingface.co/settings/tokens. "
            "Then accept model terms at huggingface.co/mistralai/Mistral-7B-Instruct-v0.3"
        ) if not token_valid else "✅ HuggingFace configured and responding",
    }
