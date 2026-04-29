#!/usr/bin/env python3
"""
scripts/probe_gtfs_rt.py
─────────────────────────
Run this ONCE to identify the correct GTFS-RT endpoint for Bloomington Transit.
This is the first thing to run after setting up the project, before starting the server.

Usage:
    cd backend
    python scripts/probe_gtfs_rt.py

It will print the URL to put in your .env as GTFS_RT_VEHICLE_URL.
"""

import asyncio
import httpx
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NOTE: These are candidate URLs based on common GTFS-RT hosting patterns.
# Bloomington Transit uses a third-party backend (likely Swiftly or TransLoc).
# If none work, visit https://bloomingtontransit.com/gtfs/ and look for
# any links to realtime feeds, or contact BT at info@bloomingtontransit.com.

VEHICLE_POSITION_CANDIDATES = [
    # Standard GTFS-RT paths
    "https://bloomingtontransit.com/gtfs/vehiclepositions.pb",
    "https://bloomingtontransit.com/gtfs/realtime/vehiclepositions.pb",
    "https://bloomingtontransit.com/realtime/vehiclepositions.pb",

    # Swiftly-hosted (common for mid-size transit agencies)
    "https://data.swiftly.com/feed/bloomingtontransit/vehiclepositions",
    "https://gtfs-rt.swiftly.com/bloomingtontransit/vehiclepositions.pb",

    # TransLoc (common for university transit)
    "https://feeds.transloc.com/3/vehicle_statuses.pb",

    # IU's own transit feed
    "https://iubus.indiana.edu/gtfs/vehiclepositions.pb",
    "https://rtpi.iu.edu/gtfs-rt/vehiclepositions",
]

TRIP_UPDATE_CANDIDATES = [
    "https://bloomingtontransit.com/gtfs/tripupdates.pb",
    "https://bloomingtontransit.com/gtfs/realtime/tripupdates.pb",
    "https://bloomingtontransit.com/realtime/tripupdates.pb",
]


async def probe(url: str, client: httpx.AsyncClient) -> dict:
    try:
        resp = await client.get(
            url,
            headers={"Accept": "application/x-protobuf, application/octet-stream, */*"},
            timeout=8.0,
        )
        if resp.status_code == 200:
            # Try to parse as protobuf
            try:
                from google.transit import gtfs_realtime_pb2
                feed = gtfs_realtime_pb2.FeedMessage()
                feed.ParseFromString(resp.content)
                n = len(feed.entity)
                has_vehicles = any(e.HasField("vehicle") for e in feed.entity)
                has_trips = any(e.HasField("trip_update") for e in feed.entity)
                return {
                    "url": url, "status": 200, "ok": True,
                    "entities": n, "has_vehicles": has_vehicles, "has_trips": has_trips,
                    "bytes": len(resp.content),
                }
            except Exception as parse_err:
                return {
                    "url": url, "status": 200, "ok": False,
                    "note": f"200 but protobuf parse failed: {parse_err}",
                    "content_preview": resp.content[:80].hex(),
                }
        else:
            return {"url": url, "status": resp.status_code, "ok": False}
    except httpx.ConnectError:
        return {"url": url, "status": None, "ok": False, "note": "Connection refused / DNS fail"}
    except Exception as e:
        return {"url": url, "status": None, "ok": False, "note": str(e)}


async def main():
    print("\n🔍 Probing Bloomington Transit GTFS-RT endpoints...\n")
    print("=" * 70)

    async with httpx.AsyncClient() as client:
        # Probe vehicle positions
        print("\n📍 VEHICLE POSITIONS endpoints:")
        vp_results = await asyncio.gather(*[probe(u, client) for u in VEHICLE_POSITION_CANDIDATES])
        vp_working = []
        for r in vp_results:
            icon = "✅" if r["ok"] else "❌"
            if r["ok"]:
                print(f"  {icon} {r['url']}")
                print(f"      {r['entities']} entities | vehicles={r['has_vehicles']} | {r['bytes']} bytes")
                vp_working.append(r)
            else:
                note = r.get("note", f"HTTP {r.get('status', 'error')}")
                print(f"  {icon} {r['url']}")
                print(f"      {note}")

        # Probe trip updates
        print("\n📋 TRIP UPDATES endpoints:")
        tu_results = await asyncio.gather(*[probe(u, client) for u in TRIP_UPDATE_CANDIDATES])
        tu_working = []
        for r in tu_results:
            icon = "✅" if r["ok"] else "❌"
            if r["ok"]:
                print(f"  {icon} {r['url']}")
                print(f"      {r['entities']} entities | trips={r['has_trips']} | {r['bytes']} bytes")
                tu_working.append(r)
            else:
                note = r.get("note", f"HTTP {r.get('status', 'error')}")
                print(f"  {icon} {r['url']}")
                print(f"      {note}")

    print("\n" + "=" * 70)
    if vp_working:
        best = vp_working[0]
        print(f"\n✅ RECOMMENDATION — add to your .env:")
        print(f"   GTFS_RT_VEHICLE_URL={best['url']}")
        if tu_working:
            print(f"   GTFS_RT_TRIP_UPDATES_URL={tu_working[0]['url']}")
    else:
        print("\n⚠️  No working GTFS-RT endpoint found.")
        print("   Steps to resolve:")
        print("   1. Visit https://bloomingtontransit.com/gtfs/ and check for RT feed links")
        print("   2. Email Bloomington Transit: info@bloomingtontransit.com")
        print("   3. Check if IU Campus Bus has a separate feed: https://iubus.indiana.edu")
        print("   4. In the meantime, the app will run with a position-simulation fallback.")
    print()


if __name__ == "__main__":
    asyncio.run(main())
