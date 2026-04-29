"""
app/api/travel_agent.py
────────────────────────
LLM travel agent using HuggingFace Inference API (free, open-source models).

LLM choice rationale:
  PRIMARY: HuggingFace Inference API — Mistral-7B-Instruct
    - Completely free with a HuggingFace account (no credit card)
    - Mistral-7B-Instruct is excellent at structured Q&A and reasoning
    - Fully open-source weights (Apache 2.0 license)
    - HF free tier: ~1000 requests/day, up to 32k tokens
    - Sign up at huggingface.co, get token from huggingface.co/settings/tokens
    - Alternative models: meta-llama/Llama-3.2-3B-Instruct (smaller/faster),
      Qwen/Qwen2.5-7B-Instruct (strong multilingual)

  FALLBACK: Rule-based response (always works, no model needed)
    - Returns structured text built from schedule data directly
    - Activates automatically if HF_TOKEN is missing or rate-limited

POST /api/travel-agent/chat
GET  /api/travel-agent/status  — check which backend is active
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

# ── HuggingFace Inference API config ─────────────────────────────────────────
# Free with a HuggingFace account. Get token at huggingface.co/settings/tokens
# Add HF_TOKEN=hf_... to backend/.env

# Model: Mistral-7B-Instruct — strong instruction following, fast, Apache 2.0
HF_MODEL = os.environ.get(
    "HF_MODEL",
    "mistralai/Mistral-7B-Instruct-v0.3"
)
# HuggingFace Inference API endpoint
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


class ChatResponse(BaseModel):
    reply: str
    model_used: str
    context_used: dict


# ── Context builder ───────────────────────────────────────────────────────────

async def _get_transit_context(db: AsyncSession, query: str) -> dict:
    now = datetime.now()
    context = {
        "current_time":    now.strftime("%I:%M %p"),
        "current_time_24": now.strftime("%H:%M:%S"),
        "day_of_week":     now.strftime("%A"),
        "date":            now.strftime("%B %d, %Y"),
    }

    weather = get_current_weather()
    if weather:
        temp_f = round(weather.get("temperature_c", 20) * 9/5 + 32)
        context["weather"] = {
            "temp_f": temp_f,
            "conditions": ("Snow" if weather.get("is_snowing")
                           else "Rain" if weather.get("is_raining")
                           else "Clear"),
            "severity": round(weather.get("weather_severity", 0), 2),
        }

    positions = await get_current_positions()
    context["live_buses"] = [
        {"vehicle_id": p["vehicle_id"], "route_id": p["route_id"],
         "lat": round(p["lat"], 4), "lng": round(p["lng"], 4)}
        for p in positions[:15]
    ]

    routes_result = await db.execute(text(
        "SELECT route_id, route_short_name, route_long_name FROM routes ORDER BY route_short_name"
    ))
    context["routes"] = [
        {"id": r[0], "short": r[1], "name": r[2]}
        for r in routes_result.fetchall()
    ]

    buildings_mentioned = _extract_buildings(query)
    nearby_stops_info = []
    for building_name, (blat, blng) in buildings_mentioned.items():
        stops_result = await db.execute(text("""
            SELECT stop_id, stop_name, stop_lat, stop_lon,
                   (ABS(stop_lat - :lat) + ABS(stop_lon - :lng)) as dist
            FROM stops ORDER BY dist ASC LIMIT 3
        """), {"lat": blat, "lng": blng})
        for row in stops_result.fetchall():
            nearby_stops_info.append({
                "near": building_name, "stop_id": row[0],
                "stop_name": row[1], "lat": row[2], "lng": row[3],
            })
    context["stops_near_mentioned_places"] = nearby_stops_info

    if nearby_stops_info:
        stop_ids = list({s["stop_id"] for s in nearby_stops_info})
        t_now = now.strftime("%H:%M:%S")
        t_end = (now + timedelta(minutes=90)).strftime("%H:%M:%S")
        # SQLite doesn't support tuple binding — build IN clause manually
        placeholders = ",".join(f"'{sid}'" for sid in stop_ids)
        arrivals_result = await db.execute(text(f"""
            SELECT st.stop_id, st.arrival_time, t.route_id,
                   r.route_short_name, r.route_long_name, t.trip_headsign
            FROM stop_times st
            JOIN trips t ON st.trip_id = t.trip_id
            JOIN routes r ON t.route_id = r.route_id
            WHERE st.stop_id IN ({placeholders})
              AND st.arrival_time >= :t_now
              AND st.arrival_time <= :t_end
            ORDER BY st.stop_id, st.arrival_time LIMIT 30
        """), {"t_now": t_now, "t_end": t_end})
        context["upcoming_arrivals"] = [
            {"stop_id": r[0], "arrival": r[1], "route": r[2],
             "route_short": r[3], "route_name": r[4], "headsign": r[5]}
            for r in arrivals_result.fetchall()
        ]
    else:
        context["upcoming_arrivals"] = []

    t_start = now.strftime("%H:%M:%S")
    t_end2  = (now + timedelta(minutes=20)).strftime("%H:%M:%S")
    releasing_result = await db.execute(text("""
        SELECT cs.building_code, cs.end_time, SUM(cs.enrollment) as total
        FROM class_sections cs
        WHERE cs.is_in_person=1 AND cs.end_time>=:ts AND cs.end_time<=:te
        GROUP BY cs.building_code, cs.end_time ORDER BY total DESC LIMIT 6
    """), {"ts": t_start, "te": t_end2})
    rows = releasing_result.fetchall()
    if rows:
        context["classes_releasing_soon"] = [
            {"building": r[0], "at": str(r[1]), "students": r[2]}
            for r in rows
        ]

    return context


BUILDING_LOCATIONS: dict[str, tuple[float, float]] = {
    "ballantine":    (39.16860, -86.52250),
    "luddy":         (39.17330, -86.52020),
    "informatics":   (39.17330, -86.52020),
    "spea":          (39.17230, -86.51390),
    "imu":           (39.16790, -86.52440),
    "memorial union":(39.16790, -86.52440),
    "jordan":        (39.17180, -86.52470),
    "chemistry":     (39.17000, -86.52050),
    "psychology":    (39.17060, -86.51960),
    "business":      (39.16950, -86.51500),
    "hodge":         (39.16950, -86.51500),
    "law":           (39.16940, -86.52200),
    "music":         (39.17520, -86.51730),
    "musical arts":  (39.17520, -86.51730),
    "wylie":         (39.16780, -86.52200),
    "woodburn":      (39.16780, -86.52370),
    "franklin":      (39.16650, -86.52250),
    "swain":         (39.17200, -86.51810),
    "wells library": (39.17090, -86.52280),
    "library":       (39.17090, -86.52280),
    "sample gates":  (39.16880, -86.52360),
    "kirkwood":      (39.16800, -86.52100),
    "rawles":        (39.16690, -86.52220),
    "lindley":       (39.16620, -86.52340),
    "eigenmann":     (39.16320, -86.51170),
    "teter":         (39.16600, -86.51800),
    "assembly hall": (39.18400, -86.52570),
    "stadium":       (39.18320, -86.52440),
    "atwater":       (39.16560, -86.51960),
    "sycamore":      (39.16870, -86.52190),
    "morrison":      (39.16780, -86.52440),
    "fine arts":     (39.17510, -86.51660),
    "global":        (39.17040, -86.51480),
    "rawles":        (39.16690, -86.52220),
}


def _extract_buildings(query: str) -> dict[str, tuple]:
    q = query.lower()
    return {name: coords for name, coords in BUILDING_LOCATIONS.items() if name in q}


def _build_prompt(context: dict) -> str:
    """Single prompt string for Ollama (no separate system field in some models)."""
    arrivals_text = ""
    for a in context.get("upcoming_arrivals", []):
        arrivals_text += f"  - Route {a['route_short']} ({a['route_name']}) at stop {a['stop_id']} arrives {a['arrival']}\n"
    if not arrivals_text:
        arrivals_text = "  No schedule data found for mentioned locations.\n"

    crowd_text = ""
    for c in context.get("classes_releasing_soon", []):
        crowd_text += f"  - {c['students']} students releasing from {c['building']} at {c['at']}\n"
    if not crowd_text:
        crowd_text = "  No major class releases in next 20 min.\n"

    weather = context.get("weather", {})
    weather_text = f"{weather.get('conditions','Clear')}, {weather.get('temp_f',70)}°F"

    stops_text = ""
    for s in context.get("stops_near_mentioned_places", []):
        stops_text += f"  - Near {s['near']}: stop '{s['stop_name']}' (ID {s['stop_id']})\n"
    if not stops_text:
        stops_text = "  No stops found near mentioned locations.\n"

    return f"""You are a helpful IU Bloomington transit assistant. Answer the student's question using the real data below. Be specific, concise (under 150 words), and friendly.

CURRENT TIME: {context['current_time']} on {context['day_of_week']}, {context['date']}
WEATHER: {weather_text}
LIVE BUSES: {len(context.get('live_buses', []))} vehicles currently on road

STOPS NEAR MENTIONED PLACES:
{stops_text}
UPCOMING BUS ARRIVALS:
{arrivals_text}
CROWD ALERT (classes releasing next 20 min):
{crowd_text}
INSTRUCTIONS:
- Give the specific route number, stop name, and departure time
- Include walk time (~5 min nearby, ~10 min far)
- If crowd alert is active near the stop, add 2-3 min buffer and warn about fullness
- State a confidence level (e.g. "~85% chance you'll make it")
- Format: Route | Leave by | Arrive by | Crowding | Summary

STUDENT'S QUESTION: """


# ── LLM backend: HuggingFace Inference API ───────────────────────────────────

async def _call_huggingface(prompt_prefix: str, messages: list[dict]) -> tuple[str, str]:
    """
    Call HuggingFace Inference API with Mistral-7B-Instruct.

    Free tier requires HF_TOKEN in .env — get one at:
      huggingface.co/settings/tokens (read token is enough)

    Uses the /v1/chat/completions endpoint (OpenAI-compatible)
    which HuggingFace added in 2024 for all Inference API calls.

    Rate limits on free tier:
      ~1,000 requests/day, 30,000 tokens/hour
    """
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        logger.info("HF_TOKEN not set — falling back to rule-based response")
        return "", ""

    # Build messages: system prompt + history + current user message
    full_messages = [{"role": "system", "content": prompt_prefix}] + messages

    # Use the OpenAI-compatible chat completions endpoint on HF
    chat_url = f"https://api-inference.huggingface.co/models/{HF_MODEL}/v1/chat/completions"

    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                chat_url,
                headers={
                    "Authorization": f"Bearer {hf_token}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": HF_MODEL,
                    "messages": full_messages,
                    "max_tokens": 450,
                    "temperature": 0.3,    # low = more factual, less creative
                    "stream": False,
                },
            )

            if resp.status_code == 200:
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                return text.strip(), f"hf/{HF_MODEL.split('/')[-1]}"

            elif resp.status_code == 503:
                # Model is loading (cold start on free tier) — inform user
                logger.warning("HF model loading (503) — may take 20s on first request")
                return (
                    "⏳ The AI model is warming up (this takes ~20 seconds on first use). "
                    "Please ask your question again in a moment.",
                    "hf/loading"
                )

            elif resp.status_code == 429:
                logger.warning("HF rate limit hit")
                return (
                    "⚠️ HuggingFace free tier rate limit reached for today. "
                    "Try again tomorrow, or upgrade at huggingface.co/pricing.",
                    ""
                )

            elif resp.status_code == 401:
                logger.error("HF token invalid or missing permissions")
                return (
                    "⚠️ HuggingFace token is invalid. Check HF_TOKEN in your backend/.env file. "
                    "Get a new token at huggingface.co/settings/tokens.",
                    ""
                )

            else:
                err_text = resp.text[:300]
                logger.warning(f"HF API error {resp.status_code}: {err_text}")
                return "", ""

    except httpx.TimeoutException:
        logger.warning("HF API timed out — model may be cold starting")
        return (
            "⏳ The AI took too long to respond (model cold start). Please try again in 20 seconds.",
            ""
        )
    except Exception as e:
        logger.error(f"HF API call failed: {e}")
        return "", ""


def _rule_based_response(context: dict, question: str) -> str:
    """
    Pure rule-based fallback when no LLM is available.
    Reads the schedule data and constructs a structured text answer.
    """
    arrivals = context.get("upcoming_arrivals", [])
    stops    = context.get("stops_near_mentioned_places", [])
    crowd    = context.get("classes_releasing_soon", [])
    weather  = context.get("weather", {})
    now_str  = context.get("current_time", "now")

    if not arrivals:
        return (
            f"🗺️ I found no scheduled arrivals for the places you mentioned in the next 90 minutes "
            f"(checked at {now_str}). Either the stops table isn't loaded yet, or service isn't "
            f"running on this route right now. Check the BT app at etaspot.net for live info."
        )

    # Group arrivals by stop
    by_stop = {}
    for a in arrivals:
        by_stop.setdefault(a["stop_id"], []).append(a)

    lines = [f"🗺️ Here's what I found for your trip (as of {now_str}):\n"]

    for stop_id, stop_arrivals in list(by_stop.items())[:2]:
        stop_info = next((s for s in stops if s["stop_id"] == stop_id), None)
        stop_label = stop_info["stop_name"] if stop_info else stop_id
        lines.append(f"📍 From **{stop_label}**:")
        for a in stop_arrivals[:3]:
            lines.append(f"   Route {a['route_short']} → {a['headsign'] or a['route_name']} at {a['arrival']}")

    if crowd:
        total = sum(c["students"] for c in crowd)
        lines.append(f"\n⚠️ Crowding: ~{total} students releasing from nearby buildings soon — expect fuller buses.")

    if weather.get("conditions") in ("Rain", "Snow"):
        lines.append(f"🌧️ Weather: {weather['conditions']} — leave a few minutes early.")

    lines.append(f"\n💡 Install Ollama (ollama.com) + run `ollama pull llama3.2` for AI-powered answers.")
    return "\n".join(lines)


# ── Main endpoint ─────────────────────────────────────────────────────────────

@router.post("/chat")
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """
    Travel agent chat. Tries Ollama → Groq → rule-based fallback.
    """
    context      = await _get_transit_context(db, request.message)
    prompt_prefix = _build_prompt(context)
    messages     = list(request.history) + [{"role": "user", "content": request.message}]

    # Try HuggingFace first, fall back to rule-based
    reply, model_used = await _call_huggingface(prompt_prefix, messages)

    if not reply:
        reply      = _rule_based_response(context, request.message)
        model_used = "rule-based"

    return ChatResponse(
        reply=reply,
        model_used=model_used,
        context_used={
            "routes_loaded":     len(context.get("routes", [])),
            "live_buses":        len(context.get("live_buses", [])),
            "arrivals_found":    len(context.get("upcoming_arrivals", [])),
            "buildings_detected": list(_extract_buildings(request.message).keys()),
            "weather":           context.get("weather"),
        },
    )


@router.get("/status")
async def llm_status():
    """Check which LLM backend is active and whether HF token is configured."""
    hf_token = os.environ.get("HF_TOKEN", "")
    has_token = bool(hf_token)

    # Optionally probe HF to check the token is valid
    token_valid = False
    if has_token:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                probe = await client.get(
                    "https://huggingface.co/api/whoami",
                    headers={"Authorization": f"Bearer {hf_token}"},
                )
                token_valid = probe.status_code == 200
        except Exception:
            pass

    return {
        "active_backend": "huggingface" if (has_token and token_valid) else "rule-based",
        "huggingface": {
            "configured": has_token,
            "token_valid": token_valid,
            "model": HF_MODEL,
            "endpoint": HF_API_URL,
        },
        "fallback": "rule-based (always available)",
        "setup_instructions": (
            "Add HF_TOKEN=hf_... to backend/.env — "
            "get a free read token at huggingface.co/settings/tokens"
        ) if not has_token else "✅ HuggingFace configured",
    }
