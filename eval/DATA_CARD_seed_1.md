# Data card: factory-energy-synthetic-seed-1

| Field | Value |
| --- | --- |
| Name and version | factory-energy-synthetic-seed-1, version 2.0 |
| How it was made | `python -m energy.generate`, numpy random generator, seed 1. No LLM and no real data. |
| Date range | 2026-04-01 to 2026-10-01 (exclusive), two quarters, UTC, 15-minute steps |
| Tables and rows | meter_15min 158,054; tariff 4; production 549; weather 183 |
| Meters | main plus 8 sub-meters: compressors, hvac, furnace, lighting, utilities, line_1, line_2, line_3 |
| Currency | INR per kWh |
| Emission factor | 0.71 kg CO2e per kWh (assumed grid average) |
| Privacy | Fully synthetic. No real company or personal data. |

## How the energy is built

Each asset has a legitimate load during the shift (06:00-22:00 on working days) and outside it, plus
production energy (kWh per unit by product: A 1.2, B 2.0, C 3.5), cooling energy
proportional to cooling degree-days (120 kWh per degree-day) and the waste patterns below.
The main meter carries everything plus 12 kW of unmetered load and 3% noise; sub-meters carry 1% noise.

## Controllable waste planted on purpose

- compressed-air leak, running day and night on `compressors`
- hall lighting left on after the last shift (22:00-06:00) on `lighting`
- HVAC running and cooling 06:00-22:00 on Sundays and holidays on `hvac`

## Traps and edge cases

- Tariff rises on 2026-08-15, in the middle of Q3 (peak 8.0 to 9.1, off-peak 6.0 to 6.8).
- The first tariff file has a wrong new off-peak price (7.4); the late drop corrects it.
- `line_2` reads 1.08x too high until the late drop recalibrates it.
- Product mix shifts towards product C (more kWh per unit) with no waste involved.
- Hot period 2026-09-01 to 2026-09-12 raises cooling.
- Public holiday 2026-08-15 is logged with a normal shift pattern.
- Sub-meters do not add up to the main meter (unmetered offices and canteen).
- 258 missing rows (two outages and random gaps), 200 duplicate rows, one meter reset (a negative reading on the furnace).

## Known limits

- One factory, two quarters; the factory clock is UTC.
- Intraday load shape does not change between quarters, so load shifting is not represented.
- Linear weather response.
