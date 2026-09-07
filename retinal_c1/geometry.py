"""Label-only chain geometry, full-image ownership, and synchronized sparse crops."""
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def chains(skeleton):
    s = np.asarray(skeleton, bool)
    h, w = s.shape
    coords = np.argwhere(s)
    count = len(coords)
    index = np.full(s.shape, -1, np.int32)
    index[tuple(coords.T)] = np.arange(count)
    neighbors = np.full((count, 8), -1, np.int32)
    for k, (dy, dx) in enumerate(((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1))):
        yy, xx = coords[:, 0] + dy, coords[:, 1] + dx
        valid = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        ids = np.flatnonzero(valid)
        if dy and dx:
            # A diagonal edge is redundant if a two-edge orthogonal route exists.
            ids = ids[~(s[yy[ids], coords[ids, 1]] | s[coords[ids, 0], xx[ids]])]
        neighbors[ids, k] = index[yy[ids], xx[ids]]
    degree = (neighbors >= 0).sum(1)
    regular = degree == 2
    node = ~regular
    rr, cc = np.where(neighbors >= 0)
    nn = neighbors[rr, cc]
    keep = node[rr] & node[nn]
    if node.any():
        node_ids = np.flatnonzero(node)
        remap = np.full(count, -1, np.int32); remap[node_ids] = np.arange(len(node_ids))
        graph = csr_matrix((np.ones(int(keep.sum())), (remap[rr[keep]], remap[nn[keep]])), shape=(len(node_ids), len(node_ids)))
        clusters = connected_components(graph, directed=False, return_labels=False)
    else:
        clusters = 0
    regular_neighbors = np.where(neighbors >= 0, regular[np.maximum(neighbors, 0)], False)
    ends = np.flatnonzero(regular & (regular_neighbors.sum(1) < 2))
    visited = np.zeros(count, bool)
    result = []
    for start in ends:
        if visited[start]:
            continue
        chain = []; previous = -1; current = int(start)
        while current >= 0 and not visited[current]:
            visited[current] = True; chain.append(current)
            options = [int(n) for n in neighbors[current] if n >= 0 and regular[n] and n != previous]
            previous, current = current, (options[0] if options else -1)
        result.append(coords[np.asarray(chain)])
    # Closed all-degree-two components have no unambiguous chain ends; report exclusion.
    return result, dict(skeleton_pixels=count, endpoint_pixels=int((degree == 1).sum()),
                        junction_pixels=int((degree > 2).sum()), node_pixels=int(node.sum()),
                        node_clusters=int(clusters), closed_cycle_pixels_excluded=int((regular & ~visited).sum()))


def group_pixels(owner, eligible, count):
    coords = np.argwhere(eligible & (owner >= 0))
    ids = owner[tuple(coords.T)] if len(coords) else np.empty(0, np.int32)
    order = np.argsort(ids, kind='stable')
    sizes = np.bincount(ids, minlength=count)
    return coords[order].astype(np.int16), np.r_[0, np.cumsum(sizes)].astype(np.int32)


def build(label, fov, skeleton, thin):
    label, fov, skeleton, thin = [np.asarray(x, bool) for x in (label, fov, skeleton, thin)]
    assert label.shape == fov.shape == skeleton.shape == thin.shape
    assert not (skeleton & ~(label & fov)).any()
    paths, stats = chains(skeleton)
    pieces = []; piece_chain = []; chain_ids = np.full(label.shape, -1, np.int32)
    short_pixels = margin_pixels = tail_pixels = 0
    for cid, path in enumerate(paths):
        chain_ids[tuple(path.T)] = cid
        if len(path) < 48:
            short_pixels += len(path); continue
        middle = path[8:-8]; margin_pixels += 16
        tail_pixels += len(middle) % 32
        for start in range(0, len(middle)-31, 32):
            pieces.append(middle[start:start+32]); piece_chain.append(cid)
    pos = np.asarray(pieces, np.int16).reshape(-1, 32, 2)
    count = len(pos)
    nearest = distance_transform_edt(~skeleton, return_distances=False, return_indices=True)
    owner_at_skeleton = np.full(label.shape, -1, np.int32)
    if count:
        owner_at_skeleton[pos[:,:,0], pos[:,:,1]] = np.arange(count)[:,None]
    owner = owner_at_skeleton[tuple(nearest)]
    dist = distance_transform_edt(~label)
    negative = fov & ~label & (dist >= 2) & (dist <= 6)
    bg, bg_offsets = group_pixels(owner, negative, count)
    fg, fg_offsets = group_pixels(owner, label & fov, count)
    bg_counts = np.diff(bg_offsets)
    valid = bg_counts >= 16
    covered = np.zeros(label.shape, bool)
    if valid.any():
        selected = pos[valid].reshape(-1, 2); covered[tuple(selected.T)] = True
    stats.update(chain_count=len(paths), short_chain_pixels_excluded=short_pixels,
                 chain_end_margin_pixels_excluded=margin_pixels, partition_tail_pixels_excluded=tail_pixels,
                 candidate_paths=count, valid_background_paths=int(valid.sum()),
                 thin_pixels=int(thin.sum()), covered_thin_pixels=int((thin & covered).sum()),
                 thin_coverage=float((thin & covered).sum()/thin.sum()) if thin.any() else None,
                 skeleton_coverage=float(covered.sum()/skeleton.sum()) if skeleton.any() else None)
    geometry = dict(pos=pos, bg=bg, bg_offsets=bg_offsets, fg=fg, fg_offsets=fg_offsets,
                    chain=np.asarray(piece_chain, np.int32), shape=np.asarray(label.shape, np.int32))
    return geometry, stats


def coordinates(points, geometry):
    top, left, height, width, hflip, vflip, turns = map(int, geometry)
    q = np.asarray(points, np.int32).copy(); q -= [top, left]
    if hflip: q[:,1] = width-1-q[:,1]
    if vflip: q[:,0] = height-1-q[:,0]
    for _ in range(turns):
        q = np.column_stack((width-1-q[:,1], q[:,0])); height, width = width, height
    return q


def select(geo, geometry, seed, *, limit=8, margin=8):
    top,left,height,width,*_ = map(int, geometry)
    pos=geo['pos']; rng=np.random.default_rng(seed)
    valid=((pos[:,:,0]>=top+margin)&(pos[:,:,0]<top+height-margin)&
           (pos[:,:,1]>=left+margin)&(pos[:,:,1]<left+width-margin)).all(1)
    result=[]
    for i in rng.permutation(np.flatnonzero(valid)):
        neg=geo['bg'][geo['bg_offsets'][i]:geo['bg_offsets'][i+1]]
        inside=(neg[:,0]>=top)&(neg[:,0]<top+height)&(neg[:,1]>=left)&(neg[:,1]<left+width)
        neg=neg[inside]
        if len(neg)<16:continue
        if len(neg)>128:neg=neg[rng.choice(len(neg),128,replace=False)]
        result.append(dict(id=int(i), pos=coordinates(pos[i],geometry), bg=coordinates(neg,geometry)))
        if limit is not None and len(result)>=limit:break
    return result
