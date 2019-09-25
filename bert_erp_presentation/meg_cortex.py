import numpy as np
from scipy.spatial.distance import cdist
from matplotlib.cm import ScalarMappable
import matplotlib
import matplotlib.colors


__all__ = [
    'surface_nearest_neighbor_input',
    'make_vertex_data',
    'dual_color_rgba_bytes',
    'cmap_rgba_bytes']


def surface_nearest_neighbor_input(surface, core_vertices, max_radius_mm, border_max_radius_mm=None):
    """
    Given a surface and a set of core vertices (typically dipole-vertices), this function finds for each surface vertex
    the core vertex to which it is closest. If a surface vertex is further than max_radius_mm from its closest core
    vertex, then it is not matched to any vertex. If border_max_radius_mm is specified, then it should be greater
    than max_radius_mm, and surface vertices which are further from their closest core vertex than max_radius_mm but
    no further than border_max_radius_mm from their closest core vertex, those matches will be returned separately.
    Args:
        surface: An instance of cortex.polyutils.Surface to match with core vertices
        core_vertices: Indices into surface.pts which specify the core vertices
        max_radius_mm: The maximum distance allowed between a core-vertex and a surface vetex for the two to match
        border_max_radius_mm: A distance, larger than max_radius_mm which specifies when a surface vertex should be
            considered 'on the border' of a core vertex

    Returns:
        indices_surface: The indices of surface vertices which are within max_radius_mm of a core vertex.
            A 1d array with shape (num_matches,) with values in the range [0, ..., len(surface.pts))
        indices_core: The indices into core_vertices of the nearest core vertex for each vertex in indices_surface.
            A 1d array with shape (num_matches,) with values in the bounds [0, ..., len(core_vertices))
        indices_border: The indices of surface vertices which are further than max_radius_mm from a core vertex and
            no further than border_max_radius_mm from a core vertex.
            A 1d array with shape (num_border_matches,) with values in the bounds [0, ..., len(surface.pts))
            Only returned in border_max_radius_mm is given
        indices_border_core: The indices into core_vertices of the nearest core vertex for each vertex in
            indices_border.
            A 1d array with shape (num_border_matches,) with values in the inclusive bounds [0, ..., len(core_vertices))
    """
    if border_max_radius_mm is not None:
        if border_max_radius_mm <= max_radius_mm:
            raise ValueError('border_max_radius_mm ({}) <= max_radius_mm ({})'.format(
                border_max_radius_mm, max_radius_mm))
    distances = cdist(surface.pts, surface.pts[core_vertices])
    indices_core = np.argmin(distances, axis=1)
    indices_surface = np.arange(len(surface.pts))
    indicator_radius = distances <= max_radius_mm
    if border_max_radius_mm is not None:
        indicator_border = np.logical_and(np.logical_not(indicator_radius), distances <= border_max_radius_mm)
        return (indices_surface[indicator_radius],
                indices_core[indicator_radius],
                indices_surface[indicator_border],
                indices_core[indicator_border])
    return indices_surface[indicator_radius], indices_core[indicator_radius]


def make_vertex_data(
        data,
        inverse_op,
        surfaces,
        mask=None,
        radius_mm=None,
        border_radius_mm=None,
        fill_value=np.nan,
        dtype=None):
    """
    Constructs an ndarray that has data on the entire surface from an array which has data only at the source-localized
    dipoles to pass into cortex.Vertex
    Args:
        data: The source localized data from which to create the new array. The left hemisphere should be concatenated
            with the right. An array with shape (num_dipoles, ...)
        inverse_op: The inverse-operator associated with the source-localization for this participant
        surfaces: The cortex.polyutils.Surface instances associated with this participant
        mask: An optional mask on data. Data is only taken from locations where the mask is True.
        radius_mm: If specified, the vertices within this distance of the dipole vertices will take the same value
            as the dipole vertex. This makes the dipoles appear larger in the visualization, which can help visibility
        border_radius_mm: The distance form dipoles to the edge of the border. If specified, this value should be
            larger than radius_mm. Vertices within this distance of the dipole vertices will take the same value as
            the dipole vertex, just as with radius_mm. An indicator array will be returned which indicates which
            vertices are further from the the nearest dipole than radius_mm and no further than border_radius_mm
        fill_value: The value to use for vertices which are not dipole vertices and are not within
            radius_mm/border_radius_mm of dipole vertices
        dtype: The data type of the returned array

    Returns:
        vertex_data. A 1d array of shape (num_vertices,)
        indicator_dipole. A boolean 1d array of shape (num_vertices,) which is True where a vertex is a dipole or is
            less than radius_mm from a dipole (when radius_mm is specified)
        indicator_border. A boolean 1d array of shape (num_vertices,) which is True where a vertex is less than
            border_radius_mm from a dipole and where indicator_dipole is False. Only returned when border_radius_mm
            is given.
    """

    if border_radius_mm is not None and radius_mm is None:
        raise ValueError('If border_radius_mm is given, radius_mm must also be given')

    if mask is None:
        left_mask = np.full(len(inverse_op['src'][0]['vertno']), True)
        right_mask = np.full(len(inverse_op['src'][1]['vertno']), True)
    else:
        left_mask = mask[:len(inverse_op['src'][0]['vertno'])]
        right_mask = mask[len(inverse_op['src'][0]['vertno']):]
        if len(right_mask) != len(inverse_op['src'][1]['vertno']):
            raise ValueError('mask length does not match inverse operator src vertices')

    # indices in the input data
    left_indices = np.arange(len(inverse_op['src'][0]['vertno']))[left_mask]
    right_indices = np.arange(len(inverse_op['src'][1]['vertno']))[right_mask]

    # indices in the VertexData
    left_vertices = inverse_op['src'][0]['vertno'][left_mask]
    right_vertices = inverse_op['src'][1]['vertno'][right_mask]

    vertex_data = np.full(
        (len(surfaces[0].pts) + len(surfaces[1].pts),) + data.shape[1:], fill_value=fill_value, dtype=dtype)
    indicator_dipole = np.full(vertex_data.shape, False)
    indicator_border = None

    if radius_mm is not None:
        if border_radius_mm is not None:
            left_vertices, left_indices_core_vertices, left_border_vertices, left_indices_border_vertices = \
                surface_nearest_neighbor_input(surfaces[0], left_vertices, radius_mm, border_radius_mm)
            left_indices = left_indices[left_indices_core_vertices]
            left_border_indices = left_indices[left_indices_border_vertices]

            right_vertices, right_indices_core_vertices, right_border_vertices, right_indices_border_vertices = \
                surface_nearest_neighbor_input(surfaces[1], right_vertices, radius_mm, border_radius_mm)
            right_indices = right_indices[right_indices_core_vertices]
            right_border_indices = right_indices[right_indices_border_vertices]

            indicator_border = np.full_like(indicator_dipole, False)
            indicator_border[left_border_vertices] = True
            indicator_border[right_border_vertices + len(surfaces[0].pts)] = True
            vertex_data[left_border_vertices] = data[left_border_indices]
            vertex_data[right_border_vertices + len(surfaces[0].pts)] = \
                data[right_border_indices + len(inverse_op['src'][0]['vertno'])]
        else:
            left_vertices, left_indices_core_vertices = surface_nearest_neighbor_input(
                surfaces[0], left_vertices, radius_mm)
            left_indices = left_indices[left_indices_core_vertices]
            right_vertices, right_indices_core_vertices = surface_nearest_neighbor_input(
                surfaces[1], right_vertices, radius_mm)
            right_indices = right_indices[right_indices_core_vertices]

    indicator_dipole[left_vertices] = True
    indicator_dipole[right_vertices + len(surfaces[0].pts)] = True
    vertex_data[left_vertices] = data[left_indices]
    vertex_data[right_vertices + len(surfaces[0].pts)] = data[right_indices + len(inverse_op['src'][0]['vertno'])]
    if border_radius_mm is not None:
        return vertex_data, indicator_dipole, indicator_border
    return vertex_data, indicator_dipole


def dual_color_rgba_bytes(data, indicator_color_1, indicator_color_2, colors=None, background=None):
    """
    Maps data to colors using a dual-color scheme. For each value in data, the value is first converted into a
    dual-color index by taking round(data) % (len(colors) ** 2). This index is then unraveled to give
    index_color_1, index_color_2 for each value in data. colors[index_color_1] is then used as the color wherever
    indicator_color_1 is True, colors[index_color_2] is used wherever indicator_color_2 is True (and indicator_color_1
    is False), and background is used everywhere else. When used in combination with make_vertex_data with a
    border_radius_mm specified, this can create dipole visualizations which have an inner color and a border color.
    That can be useful for plotting a large number of clusters, for example, since the clusters can be distinguished
    from each other by 2 colors instead of just 1.
    Args:
        data: The data to map to colors. If not an integer type, the data will be rounded and converted to an
            integer type
        indicator_color_1: Which points in the data should take color-1
        indicator_color_2: Which points in the data should take color-2
        colors: The color palette to draw from. The number of patterns available is len(colors) ** 2. If not specified,
            defaults to the matplotlib color cycle (i.e. TABLEAU_COLORS), which has 10 separate colors (100 patterns)
        background: The color for the background. If not specified, defaults to (0, 0, 0, 0)

    Returns:
        rgba_bytes: An array with shape data.shape + (4,) that has type uint8 with values ranging from 0-255.
    """
    if colors is None:
        colors = matplotlib.colors.TABLEAU_COLORS

    colors = np.array(list(matplotlib.colors.to_rgba(c) for c in colors))
    background = matplotlib.colors.to_rgba('none') if background is None else matplotlib.colors.to_rgba(background)

    result = np.empty(data.shape + (4,), dtype=colors.dtype)

    indicator_foreground = np.logical_or(indicator_color_1, indicator_color_2)
    c = data[indicator_foreground]
    indicator_c1 = indicator_color_1[indicator_foreground]

    if not np.issubdtype(c.dtype, np.integer):
        c = np.round(c).astype(int)
    c = np.mod(c, len(colors) * len(colors))
    c1, c2 = np.unravel_index(c, (len(colors), len(colors)))
    c = np.where(indicator_c1, c1, c2)
    result[indicator_foreground] = colors[c]
    result[np.logical_not(indicator_foreground)] = background

    return (result * 255).astype(np.uint8)


def cmap_rgba_bytes(data, indicator_foreground=None, cmap=None, vmin=None, vmax=None, background=None):
    """
    Maps data to colors using a colormap.
    Args:
        data: The data to map to colors.
        indicator_foreground: If specified, any data where indicator_foreground is False will take the background
            color (or use alpha=0 if background is None). np.isnan(data) is also put in the background
        cmap: The colormap to use. If None, the default matplotlib colormap will be used
        vmin: The minimum value for colormap normalization. If None, the value is determined from the data.
        vmax: The maximum value for colormap normalization. If None, the value is determined from the data.
        background: The color for data which is nan or where indicator_foreground is False. If None, defaults to
            (0, 0, 0, 0)

    Returns:

    """
    if indicator_foreground is not None:
        indicator_foreground = np.logical_and(indicator_foreground, np.logical_not(np.isnan(data)))
    else:
        indicator_foreground = np.logical_not(np.isnan(data))

    if background is None:
        background = 'none'

    background = (255 * np.array(matplotlib.colors.to_rgba(background))).astype(np.uint8)

    mappable = ScalarMappable(norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True), cmap=cmap)

    min_value = 0 if np.count_nonzero(indicator_foreground) == 0 else np.nanmin(data)
    rgba = mappable.to_rgba(np.where(indicator_foreground, data, min_value), bytes=True)
    rgba = np.where(np.expand_dims(indicator_foreground, 1), rgba, np.expand_dims(background, 0))

    return rgba
