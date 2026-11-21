const uint index = thread_position_in_grid.x;
if (index >= uint(n_items)) return;
result[index] = inkling_capped_value(
    projected[2 * index], projected[2 * index + 1], route_weights[index / WIDTH]);
