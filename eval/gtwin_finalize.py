"""Round-2 sonucundan GT montaji: union-find + zaman-ortusme korumasi.

Girdi: gtwin_adj_result2.json (round-2 purity+pair_results)
       gtwin_units.json, gtwin_unit_intervals.json, gtwin_spurious_units.json
Cikti: gtwin_gt.json  {tid2ident, spurious_tids, excluded_units, identities,
                       ident_cards}  + gtwin_identity_groups.json (kart uretimi icin)
"""
from __future__ import annotations
import json


def main():
    res = json.load(open('scratchpad/gtwin_adj_result2.json'))
    units = json.load(open('scratchpad/gtwin_units.json'))
    intervals = json.load(open('scratchpad/gtwin_unit_intervals.json'))
    spur = json.load(open('scratchpad/gtwin_spurious_units.json'))

    pur = res['purity']
    ok_units = [u for u in units
                if pur.get(u, {}).get('pure') and pur.get(u, {}).get('is_player')]
    excluded = [u for u in units if u not in ok_units]

    parent = {}

    def find(k):
        while parent.get(k, k) != k:
            k = parent[k]
        return k

    members = {u: [u] for u in ok_units}

    def overlaps(ra, rb, tol=0.15):
        for ma in members[ra]:
            for mb in members[rb]:
                for a0, a1 in intervals[ma]:
                    for b0, b1 in intervals[mb]:
                        if min(a1, b1) - max(a0, b0) > tol:
                            return True
        return False

    accepted = sorted([r for r in res['pair_results'] if r['final'] == 'same'
                       and r['a'] in members and r['b'] in members],
                      key=lambda r: r['cos'])
    rej_overlap = 0
    for r in accepted:
        ra, rb = find(r['a']), find(r['b'])
        if ra == rb:
            continue
        if overlaps(ra, rb):
            rej_overlap += 1
            continue
        parent[rb] = ra
        members[ra] = members[ra] + members[rb]
        del members[rb]

    idents = {f"I{k}": sorted(mem) for k, (root, mem) in
              enumerate(sorted(members.items(),
                               key=lambda x: -sum(len(units[u]) for u in x[1])))}
    tid2ident = {}
    for iname, mem in idents.items():
        for u in mem:
            for t in units[u]:
                tid2ident[str(t)] = iname
    spurious_tids = sorted({t for s in spur for t in s['tids']} |
                           {t for u in excluded
                            if not pur.get(u, {}).get('is_player', True)
                            for t in units[u]})
    excluded_tids = sorted({t for u in excluded for t in units[u]})

    gt = {'tid2ident': tid2ident,
          'identities': idents,
          'spurious_tids': spurious_tids,
          'excluded_tids': excluded_tids,
          'excluded_units': excluded,
          'rejected_by_overlap': rej_overlap,
          'meta': {'n_identity': len(idents),
                   'n_unit_ok': len(ok_units),
                   'n_unit_excluded': len(excluded),
                   'source': 'round2 agent adjudication wf_c69ec887'}}
    json.dump(gt, open('scratchpad/gtwin_gt.json', 'w'), indent=1)
    # kimlik kartlari icin gruplar (>=2 birimli kimlikler doğrulanacak)
    ig = {iname: sorted({t for u in mem for t in units[u]})
          for iname, mem in idents.items()}
    json.dump(ig, open('scratchpad/gtwin_identity_groups.json', 'w'), indent=1)
    multi = {i: m for i, m in idents.items() if len(m) >= 2}
    print(f"{len(idents)} kimlik ({len(multi)} cok-birimli, dogrulama gerekli), "
          f"{len(excluded)} birim haric, overlap-red {rej_overlap}")
    print('kimlik boyutlari (det):',
          {i: sum(len(units[u]) for u in m) for i, m in list(idents.items())[:20]})


if __name__ == '__main__':
    main()
