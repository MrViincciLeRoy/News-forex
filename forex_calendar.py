"""
Forex Event Calendar - Smart Business Day Logic
Covers: USD, EUR, GBP, JPY, CAD, AUD, CHF, NZD
"""

import json
from datetime import datetime
from calendar import monthrange
from pathlib import Path
from typing import Dict, List, Optional


IMPORTANCE_EMOJI = {
    'EXTREME': '🔴',
    'HIGH':    '🟠',
    'MEDIUM':  '🟡',
    'LOW':     '⚪',
}

CURRENCY_FLAG = {
    'USD': '🇺🇸', 'EUR': '🇪🇺', 'GBP': '🇬🇧',
    'JPY': '🇯🇵', 'CAD': '🇨🇦', 'AUD': '🇦🇺',
    'CHF': '🇨🇭', 'NZD': '🇳🇿',
}


class ForexCalendar:

    def __init__(self, output_dir: str = 'forex_events'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Public Interface                                                    #
    # ------------------------------------------------------------------ #

    def fetch_calendar(
        self,
        start_date: str,
        end_date: str,
        importance_filter: Optional[List[str]] = None,
        currency_filter: Optional[List[str]] = None,
    ) -> List[Dict]:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end   = datetime.strptime(end_date,   '%Y-%m-%d')

        events = self._collect_events(start, end)

        if importance_filter:
            events = [e for e in events if e['importance'] in importance_filter]
        if currency_filter:
            events = [e for e in events if e['currency'] in currency_filter]

        events.sort(key=lambda x: (x['date'], x['time']))
        return events

    def generate_monthly_calendar(self, year: int, month: int) -> Dict:
        num_days   = monthrange(year, month)[1]
        start_date = f"{year}-{month:02d}-01"
        end_date   = f"{year}-{month:02d}-{num_days:02d}"

        events = self.fetch_calendar(start_date, end_date)

        calendar = {
            'year':         year,
            'month':        month,
            'month_name':   datetime(year, month, 1).strftime('%B %Y'),
            'total_events': len(events),
            'by_importance': {
                'EXTREME': sum(1 for e in events if e['importance'] == 'EXTREME'),
                'HIGH':    sum(1 for e in events if e['importance'] == 'HIGH'),
                'MEDIUM':  sum(1 for e in events if e['importance'] == 'MEDIUM'),
                'LOW':     sum(1 for e in events if e['importance'] == 'LOW'),
            },
            'events': events,
        }

        self._print_calendar(calendar)
        self._save(calendar, f"forex_calendar_{year}_{month:02d}.json")
        return calendar

    # ------------------------------------------------------------------ #
    #  Event Generation                                                   #
    # ------------------------------------------------------------------ #

    def _collect_events(self, start: datetime, end: datetime) -> List[Dict]:
        events = []
        cur = datetime(start.year, start.month, 1)

        while cur <= end:
            for event in self._month_events(cur.year, cur.month):
                d = datetime.strptime(event['date'], '%Y-%m-%d')
                if start <= d <= end:
                    events.append(event)

            cur = datetime(cur.year + 1, 1, 1) if cur.month == 12 \
                  else datetime(cur.year, cur.month + 1, 1)

        return events

    def _month_events(self, year: int, month: int) -> List[Dict]:
        num_days = monthrange(year, month)[1]

        def date(day: int) -> str:
            return f"{year}-{month:02d}-{day:02d}"

        def is_weekday(day: int) -> bool:
            return datetime(year, month, day).weekday() < 5

        def next_biz(day: int) -> Optional[int]:
            while day <= num_days:
                if is_weekday(day):
                    return day
                day += 1
            return None

        def nth_weekday(n: int, wd: int) -> Optional[int]:
            """nth occurrence of weekday (0=Mon … 6=Sun)"""
            count, day = 0, 1
            while day <= num_days:
                if datetime(year, month, day).weekday() == wd:
                    count += 1
                    if count == n:
                        return day
                day += 1
            return None

        def all_weekdays(wd: int) -> List[int]:
            return [d for d in range(1, num_days + 1)
                    if datetime(year, month, d).weekday() == wd]

        def ev(day, time, name, country, currency, importance):
            if day is None:
                return None
            return dict(date=date(day), time=time, name=name,
                        country=country, currency=currency,
                        importance=importance)

        events = []
        add = lambda e: events.append(e) if e else None

        # ── USD ──────────────────────────────────────────────────────── #
        add(ev(nth_weekday(1, 4), '08:30', 'Non-Farm Payrolls',       'United States', 'USD', 'EXTREME'))
        add(ev(next_biz(12),      '08:30', 'Consumer Price Index',     'United States', 'USD', 'EXTREME'))

        if month in [1, 3, 5, 6, 7, 9, 11, 12]:
            add(ev(nth_weekday(2, 2), '14:00', 'FOMC Rate Decision',  'United States', 'USD', 'EXTREME'))

        add(ev(next_biz(1),  '10:00', 'ISM Manufacturing PMI',        'United States', 'USD', 'HIGH'))
        add(ev(next_biz(14), '08:30', 'Retail Sales',                 'United States', 'USD', 'HIGH'))
        add(ev(next_biz(13), '08:30', 'Producer Price Index',         'United States', 'USD', 'HIGH'))
        add(ev(next_biz(15), '09:15', 'Industrial Production',        'United States', 'USD', 'MEDIUM'))
        add(ev(next_biz(25), '10:00', 'Consumer Confidence',          'United States', 'USD', 'MEDIUM'))
        add(ev(next_biz(17), '10:00', 'Existing Home Sales',          'United States', 'USD', 'MEDIUM'))

        if month in [1, 4, 7, 10]:
            add(ev(next_biz(27), '08:30', 'Advance GDP',              'United States', 'USD', 'EXTREME'))
        if month in [2, 5, 8, 11]:
            add(ev(next_biz(26), '08:30', 'GDP (Second Estimate)',     'United States', 'USD', 'HIGH'))
        if month in [3, 6, 9, 12]:
            add(ev(next_biz(26), '08:30', 'GDP (Final)',               'United States', 'USD', 'HIGH'))

        # Weekly Jobless Claims – every Thursday
        for d in all_weekdays(3):
            add(ev(d, '08:30', 'Initial Jobless Claims',              'United States', 'USD', 'MEDIUM'))

        # ── EUR ──────────────────────────────────────────────────────── #
        if month in [1, 3, 4, 6, 7, 9, 10, 12]:
            add(ev(nth_weekday(2, 3), '07:45', 'ECB Rate Decision',   'Euro Area',     'EUR', 'EXTREME'))

        add(ev(next_biz(1),  '04:00', 'German Manufacturing PMI',    'Germany',       'EUR', 'HIGH'))
        add(ev(next_biz(16), '05:00', 'Eurozone CPI (Flash)',        'Euro Area',     'EUR', 'HIGH'))
        add(ev(next_biz(23), '05:00', 'German Ifo Business Climate', 'Germany',       'EUR', 'MEDIUM'))

        if month in [1, 4, 7, 10]:
            add(ev(next_biz(28), '05:00', 'Eurozone GDP (Flash)',    'Euro Area',     'EUR', 'HIGH'))

        # ── GBP ──────────────────────────────────────────────────────── #
        if month in [2, 3, 5, 6, 8, 9, 11, 12]:
            add(ev(nth_weekday(1, 3), '12:00', 'BoE Rate Decision',  'United Kingdom', 'GBP', 'EXTREME'))

        add(ev(next_biz(16), '02:00', 'UK CPI',                     'United Kingdom', 'GBP', 'HIGH'))
        add(ev(next_biz(10), '02:00', 'UK GDP (Monthly)',            'United Kingdom', 'GBP', 'HIGH'))
        add(ev(next_biz(22), '02:00', 'UK Retail Sales',            'United Kingdom', 'GBP', 'MEDIUM'))
        add(ev(next_biz(18), '02:00', 'UK Employment Change',       'United Kingdom', 'GBP', 'HIGH'))

        # ── JPY ──────────────────────────────────────────────────────── #
        add(ev(nth_weekday(1, 2), '23:50', 'BoJ Rate Decision',     'Japan',          'JPY', 'EXTREME'))
        add(ev(next_biz(27),      '23:30', 'Tokyo CPI',             'Japan',          'JPY', 'HIGH'))
        add(ev(next_biz(20),      '23:50', 'Japan Trade Balance',   'Japan',          'JPY', 'MEDIUM'))

        if month in [2, 5, 8, 11]:
            add(ev(next_biz(15), '23:50', 'Japan GDP',              'Japan',          'JPY', 'HIGH'))

        # ── CAD ──────────────────────────────────────────────────────── #
        if month in [1, 3, 4, 6, 7, 9, 10, 12]:
            add(ev(nth_weekday(2, 2), '10:00', 'BoC Rate Decision', 'Canada',         'CAD', 'EXTREME'))

        add(ev(nth_weekday(1, 4), '08:30', 'Canada Employment',     'Canada',         'CAD', 'HIGH'))
        add(ev(next_biz(18),      '08:30', 'Canada CPI',            'Canada',         'CAD', 'HIGH'))
        add(ev(next_biz(15),      '08:30', 'Canada Retail Sales',   'Canada',         'CAD', 'MEDIUM'))

        # ── AUD ──────────────────────────────────────────────────────── #
        if month in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            add(ev(nth_weekday(1, 1), '03:30', 'RBA Rate Decision', 'Australia',      'AUD', 'EXTREME'))

        add(ev(next_biz(15), '21:30', 'Australia Employment',       'Australia',      'AUD', 'HIGH'))

        if month in [1, 4, 7, 10]:
            add(ev(next_biz(25), '21:30', 'Australia CPI',          'Australia',      'AUD', 'HIGH'))

        # ── CHF ──────────────────────────────────────────────────────── #
        if month in [3, 6, 9, 12]:
            add(ev(nth_weekday(2, 3), '08:30', 'SNB Rate Decision', 'Switzerland',    'CHF', 'EXTREME'))

        add(ev(next_biz(8), '03:30', 'Switzerland CPI',             'Switzerland',    'CHF', 'MEDIUM'))

        # ── NZD ──────────────────────────────────────────────────────── #
        if month in [2, 4, 5, 7, 8, 10, 11]:
            add(ev(nth_weekday(1, 2), '21:00', 'RBNZ Rate Decision','New Zealand',    'NZD', 'EXTREME'))

        add(ev(next_biz(16), '21:45', 'NZ CPI',                    'New Zealand',    'NZD', 'HIGH'))

        return events

    # ------------------------------------------------------------------ #
    #  Display                                                            #
    # ------------------------------------------------------------------ #

    def _print_calendar(self, calendar: Dict):
        W = 80
        print(f"\n{'═'*W}")
        print(f"  📅  {calendar['month_name'].upper()} — FOREX ECONOMIC CALENDAR")
        print(f"{'═'*W}")
        print(f"  Total: {calendar['total_events']} events  |  "
              f"🔴 Extreme: {calendar['by_importance']['EXTREME']}  "
              f"🟠 High: {calendar['by_importance']['HIGH']}  "
              f"🟡 Medium: {calendar['by_importance']['MEDIUM']}")
        print(f"{'─'*W}\n")

        current_date = None
        for e in calendar['events']:
            if e['date'] != current_date:
                current_date = e['date']
                d = datetime.strptime(e['date'], '%Y-%m-%d')
                print(f"  📆  {d.strftime('%A, %B %d, %Y')}")

            flag  = CURRENCY_FLAG.get(e['currency'], '  ')
            emoji = IMPORTANCE_EMOJI[e['importance']]

            print(f"       {emoji} {e['time']}  {flag} {e['currency']:3s}  {e['name']}")
            print(f"             {e['country']} · {e['importance']}")

        print(f"\n{'═'*W}")

    # ------------------------------------------------------------------ #
    #  Persistence                                                        #
    # ------------------------------------------------------------------ #

    def _save(self, data: Dict, filename: str):
        fp = self.output_dir / filename
        fp.write_text(json.dumps(data, indent=2))
        print(f"  💾  Saved → {fp}\n")


# ------------------------------------------------------------------ #
#  Entry Point                                                        #
# ------------------------------------------------------------------ #

def main():
    cal = ForexCalendar()
    now = datetime.now()

    # Current month — all events
    cal.generate_monthly_calendar(now.year, now.month)

    # Custom range — extreme + high only
    print("FILTERED VIEW: Extreme & High impact — Feb 2026")
    print('─' * 60)
    events = cal.fetch_calendar(
        '2026-02-01', '2026-02-28',
        importance_filter=['EXTREME', 'HIGH'],
    )
    for e in events:
        flag  = CURRENCY_FLAG.get(e['currency'], '  ')
        emoji = IMPORTANCE_EMOJI[e['importance']]
        print(f"  {e['date']} {e['time']}  {emoji} {flag} {e['currency']}  {e['name']}")


if __name__ == '__main__':
    main()
