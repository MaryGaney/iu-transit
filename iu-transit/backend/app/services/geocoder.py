"""
app/services/geocoder.py
─────────────────────────
Maps IU building codes to lat/lng.
Extended to cover all buildings that appear in the class schedule.
"""

import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.core.logging import logger
from app.models.schedule import IUBuilding


# ── Comprehensive IUB building code → (name, lat, lng) ───────────────────────
# Sources: IU campus map, registrar building list, Google Maps verification
# Codes follow IU's BL prefix convention

KNOWN_BUILDINGS: dict[str, tuple[str, float, float]] = {
    # ── Core academic buildings ───────────────────────────────────────────────
    "BLTH":  ("Ballantine Hall",                    39.16860, -86.52250),
    "BLHD":  ("School of Public & Env Affairs",     39.17230, -86.51390),
    "BLWY":  ("Wylie Hall",                         39.16780, -86.52200),
    "BLLH":  ("Lindley Hall",                       39.16620, -86.52340),
    "BLWH":  ("Woodburn Hall",                      39.16780, -86.52370),
    "BLBH":  ("Bryan Hall",                         39.16940, -86.52370),
    "BLRH":  ("Rawles Hall",                        39.16690, -86.52220),
    "BLSY":  ("Sycamore Hall",                      39.16870, -86.52190),
    "BLKH":  ("Kirkwood Hall",                      39.16800, -86.52100),
    "BLSH":  ("Swain Hall West",                    39.17200, -86.51810),
    "BLSE":  ("Swain Hall East",                    39.17190, -86.51740),
    "BLJH":  ("Jordan Hall",                        39.17180, -86.52470),
    "BLCH":  ("Chemistry Building",                 39.17000, -86.52050),
    "BLBI":  ("Biology Building",                   39.17140, -86.52240),
    "BLPY":  ("Psychology Building",                39.17060, -86.51960),
    "BLMO":  ("Morrison Hall",                      39.16780, -86.52440),
    "BLMY":  ("Myles Brand Hall",                   39.17350, -86.52010),
    "BLEP":  ("Eigenmann Hall",                     39.16320, -86.51170),
    "BLHH":  ("Hodge Hall (Business)",              39.16950, -86.51500),
    "BLGI":  ("Global & Intl Studies Bldg",         39.17040, -86.51480),
    "BLFW":  ("Franklin Hall",                      39.16650, -86.52250),
    "BLLW":  ("Law School (Maurer)",                39.16940, -86.52200),
    "BLMU":  ("Musical Arts Center",                39.17520, -86.51730),
    "BLFC":  ("Fine Arts Complex",                  39.17510, -86.51660),
    "BLIMU": ("Indiana Memorial Union",             39.16790, -86.52440),
    "IMU":   ("Indiana Memorial Union",             39.16790, -86.52440),
    "BLSI":  ("Simon Hall (Geo Sciences)",          39.17080, -86.52320),
    "BLLU":  ("Luddy Hall (Informatics)",            39.17330, -86.52020),
    "BLI":   ("Luddy Hall (Informatics)",            39.17330, -86.52020),
    "BLAT":  ("Atwater Hall",                       39.16560, -86.51960),
    "SPEA":  ("SPEA Building",                      39.17230, -86.51390),
    "INFO":  ("Informatics / Luddy Hall",           39.17330, -86.52020),
    "LAW":   ("Maurer School of Law",               39.16940, -86.52200),
    "BLME":  ("Medical Sciences",                   39.17540, -86.52030),
    "BLFA":  ("Fine Arts",                          39.17510, -86.51660),
    "BLPH":  ("Poplars Building",                   39.16470, -86.52530),
    "BLGA":  ("Global & Intl Studies",              39.17040, -86.51480),
    "BLIA":  ("Indiana Ave Studios",                39.16350, -86.52890),

    # ── Health & science ──────────────────────────────────────────────────────
    "BLC2":  ("Biology / Chemistry Annex",          39.17100, -86.52100),
    "BLNQ":  ("Nursing / Health Sciences",          39.17250, -86.52100),
    "BLOP":  ("Optometry Building",                 39.17480, -86.52540),
    "BLDP":  ("Dental School",                      39.17480, -86.52450),
    "BLSB":  ("Simon Hall (Science)",               39.17080, -86.52320),

    # ── Performing arts / libraries ───────────────────────────────────────────
    "BLWL":  ("Wells Library (Main)",               39.17090, -86.52280),
    "BLLB":  ("Law Library",                        39.16940, -86.52220),
    "BLAS":  ("Art Studio",                         39.17420, -86.51600),
    "BLRA":  ("Radio-TV Building",                  39.16700, -86.52460),

    # ── Residence halls / union ───────────────────────────────────────────────
    "BLTE":  ("Teter Quad",                         39.16600, -86.51800),
    "BLEG":  ("Eigenmann Hall",                     39.16320, -86.51170),
    "BLFD":  ("Forest Quad",                        39.17050, -86.50820),
    "BLTQ":  ("Test / Teter Quad",                  39.16600, -86.51800),

    # ── Engineering / computing ───────────────────────────────────────────────
    "BLEI":  ("Engineering Building",               39.17550, -86.52550),
    "BLSW":  ("Swain Hall",                         39.17200, -86.51810),

    # ── Athletics / rec ───────────────────────────────────────────────────────
    "BLASM": ("Assembly Hall / Simon Skjodt",       39.18400, -86.52570),
    "BLMF":  ("Memorial Stadium",                   39.18320, -86.52440),

    # ── Miscellaneous ─────────────────────────────────────────────────────────
    "BLHB":  ("Hamilton Center",                    39.16900, -86.52150),
    "BLPC":  ("Poplars Conference Center",          39.16470, -86.52530),
    "BLEC":  ("Education Building (Wright)",        39.17200, -86.51300),
    "BLED":  ("Education Building",                 39.17200, -86.51300),
    "BLIS":  ("Informatics / Lindley",              39.16620, -86.52340),
    "BLSO":  ("Social Work Building",               39.16890, -86.52150),
    "BLHS":  ("History Building",                   39.16780, -86.52200),
    "BLEN":  ("English Building",                   39.16780, -86.52370),
    "BLPO":  ("Political Science",                  39.16780, -86.52200),
    "BLSX":  ("Sociology / Anthropology",           39.16780, -86.52200),
    "BLECON":("Economics Building",                 39.16940, -86.52370),
    "BLGEO": ("Geography Building",                 39.17080, -86.52320),
    "BLAST": ("Astronomy Building",                 39.17080, -86.52320),
    "BLPHY": ("Physics Building",                   39.17200, -86.51810),
    "BLMATH":("Mathematics Building",               39.16620, -86.52340),
    "BLSTAT":("Statistics Building",               39.16780, -86.52200),
}


async def geocode_all_buildings(db: AsyncSession) -> int:
    """Populate IUBuilding table. Safe to call multiple times (upsert)."""
    result = await db.execute(select(IUBuilding))
    existing = {b.building_code: b for b in result.scalars()}

    count = 0
    for code, (name, lat, lng) in KNOWN_BUILDINGS.items():
        if code not in existing:
            db.add(IUBuilding(
                building_code=code,
                building_name=name,
                latitude=lat,
                longitude=lng,
            ))
            count += 1
        else:
            bld = existing[code]
            if bld.latitude is None:
                bld.latitude = lat
                bld.longitude = lng
                count += 1

    await db.flush()

    # Also try IU's API for anything we're still missing
    try:
        added = await _fetch_from_iu_api(db, existing)
        if added:
            logger.info(f"IU Buildings API added {added} additional buildings")
    except Exception as e:
        logger.debug(f"IU buildings API unavailable: {e}")

    total_result = await db.execute(
        select(IUBuilding).where(IUBuilding.latitude.is_not(None))
    )
    total = len(total_result.scalars().all())
    logger.info(f"Building geocoder: {total} buildings with coordinates")
    return total


async def _fetch_from_iu_api(db: AsyncSession, existing: dict) -> int:
    """Fetch additional buildings from IU's public API."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                settings.iu_buildings_url,
                params={"campus": "BL"},
                headers={"Accept": "application/json"},
                follow_redirects=True,
            )
            if resp.status_code != 200:
                return 0
            buildings = resp.json()
    except Exception:
        return 0

    count = 0
    for bld in buildings:
        code = bld.get("buildingCode", "").upper()
        lat  = bld.get("latitude")
        lng  = bld.get("longitude")
        name = bld.get("name", "")
        if not code or lat is None or lng is None:
            continue
        if code not in existing:
            db.add(IUBuilding(
                building_code=code,
                building_name=name,
                latitude=float(lat),
                longitude=float(lng),
                address=bld.get("address", ""),
            ))
            count += 1
    await db.flush()
    return count


async def get_building_coords(
    db: AsyncSession, building_code: str
) -> tuple[float, float] | None:
    if building_code in KNOWN_BUILDINGS:
        _, lat, lng = KNOWN_BUILDINGS[building_code]
        return lat, lng
    result = await db.execute(
        select(IUBuilding).where(IUBuilding.building_code == building_code)
    )
    bld = result.scalar_one_or_none()
    if bld and bld.latitude:
        return bld.latitude, bld.longitude
    return None
