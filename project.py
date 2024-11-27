import random
import numpy as np
import math

# functions for algorithms 1 & 2

def are_matrices_equal(original_mat, output_1_mat):  # Code part for testing algorithms 1 and 2
    # Check if the dimensions of the matrices are the same
    n = int(len(original_mat))
    original_mat = np.array(original_mat)
    original_mat[n - 1, n - 1] = 1
    if len(original_mat) != len(output_1_mat):
        return False

    for row in range(len(original_mat)):
        if len(original_mat[row]) != len(output_1_mat[row]):
            return False
        for col in range(len(original_mat[row])):
            if original_mat[row][col] != output_1_mat[row][col]:
                return False
    return True


def find_submatrix(matrix, n, L):
    for i in range(n - L + 1):
        for j in range(n - L + 1):
            # Check if this position is the top-left corner of an LxL submatrix of zeros
            if all(matrix[i + k][j + l] == 0 for k in range(L) for l in range(L)):
                return (i, j)  # Return the index of the top-left corner of the submatrix
    return None  # If no such submatrix is found


def replace_submatrix(input_matrix, lookup_square, init_index, L):
    for i in range(L):
        for j in range(L):
            input_matrix[init_index[0] + i][init_index[1] + j] = lookup_square[i][j]
    return input_matrix


def matrix_to_vector(matrix):
    vector = [elem for row in matrix for elem in row]
    return vector


def vector_to_matrix(string):
    L = int(np.sqrt(len(string)))
    elements = [int(char) for char in string]
    matrix = np.array(elements).reshape((L, L))
    return matrix


def vector_to_matrices(string, k):
    num_matrices = len(string) // (k * k)
    matrices = []
    for i in range(num_matrices):
        part = string[i * k * k: (i + 1) * k * k]
        elements = [int(char) for char in part]
        matrix = np.array(elements).reshape((k, k))
        matrices.append(matrix)
    return matrices


def generate_new_lookup(target_index, input_matrix, n, L):
    row_index, col_index = target_index
    target_order = row_index * n + col_index

    binary_num = bin(target_order)[2:]
    num_of_bits = math.ceil(
        math.log2(len(input_matrix) ** 2))  # Number of bits to represent the number (a general cell in the matrix)
    num_zeros = num_of_bits - len(binary_num)  # Calculate the number of zeros needed to concatenate on left side
    binary_num = '0' * num_zeros + binary_num  # Concatenate zeros to the left
    lookup_str = binary_num + '0' * (L ** 2 - len(binary_num))
    lookup_square = vector_to_matrix(lookup_str)
    return lookup_square


def check_overlap(n, L, v):
    if v[0] + L > n - L and v[1] + L > n - L:
        return True
    else:
        return False


def count_overlaps(v, L, n):
    (x1_start, y1_start, x1_end, y1_end) = (n - L, n - L, n - 1, n - 1)
    (x2_start, y2_start, x2_end, y2_end) = (v[1], v[0], v[1] + L - 1, v[0] + L - 1)

    overlap_x_start = max(x1_start, x2_start)  # Calculate the overlap boundaries
    overlap_y_start = max(y1_start, y2_start)
    overlap_x_end = min(x1_end, x2_end)
    overlap_y_end = min(y1_end, y2_end)

    overlap_count = 0
    for i in range(overlap_x_start, overlap_x_end + 1):
        for j in range(overlap_y_start, overlap_y_end + 1):
            overlap_count += 1
    return overlap_count


def extract_location(lookup_square, input_matrix, n, e, L):
    lookup_vector = matrix_to_vector(lookup_square)
    num_of_bits = math.ceil(math.log2(len(input_matrix) ** 2))  # number of bits to represent the number
    num_zeros = L ** 2 - num_of_bits  # Calculate the number of zeros needed to reduce
    binary_num = lookup_vector[:-num_zeros]  # reduce zeros from the right side
    binary_str = ''.join(map(str, binary_num))  # Convert the list of integers to a string of digits
    num = int(binary_str, 2)
    target_index = [num // n, num % n]
    v_location = target_index - e
    return v_location


def matrix_to_blocks(matrix, k):
    n, m = len(matrix), len(matrix[0])
    blocks = []
    for i in range(0, n, k):
        for j in range(0, m, k):
            block = np.array([row[j:j + k] for row in matrix[i:i + k]])
            blocks.append(block)
    return blocks


def blocks_to_matrix(blocks):
    blocks = np.array(blocks)
    block_size = blocks[0].shape[0]
    num_blocks = len(blocks)
    num_rows = int(np.ceil(np.sqrt(num_blocks)))
    num_cols = int(np.ceil(num_blocks / num_rows))
    matrix = np.zeros((num_rows * block_size, num_cols * block_size), dtype=blocks[0].dtype)

    for idx, block in enumerate(blocks):
        row_start = (idx // num_cols) * block_size
        col_start = (idx % num_cols) * block_size
        matrix[row_start:row_start + block_size, col_start:col_start + block_size] = block
    return matrix


# functions for algorithm 3 #
def remove_blocks_case1(input_matrix, k, t):
    blocks = matrix_to_blocks(input_matrix, k)
    for i in range(t):
        if blocks:
            blocks.pop()
    new_matrix = blocks_to_matrix(blocks)
    return new_matrix


def remove_blocks_case2(input_matrix, k, t):  # t = num of blocks to remove
    if t == 0:
        return input_matrix

    blocks = matrix_to_blocks(input_matrix, k)
    remaining_blocks = []
    pm_block = np.array([[1, 0], [0, 0]])

    for block in blocks:
        if not np.all(block == -1):
            remaining_blocks.append(block)

    remaining_blocks = remaining_blocks[:-t]
    remaining_blocks[-1] = pm_block
    n, m = input_matrix.shape
    total_blocks = (n // k) * (m // k)

    while len(remaining_blocks) < total_blocks:
        remaining_blocks.append(np.full((k, k), -1))
    new_matrix = blocks_to_matrix(remaining_blocks)

    return new_matrix


def pre_replacement(matrix, i, j, k, i_hat, j_hat):
    matrix_1 = matrix[i_hat * k:(i_hat + 1) * k, j_hat * k:(j_hat + 1) * k].copy()
    matrix_2 = matrix[i:i + k, j:j + k].copy()

    positions_1 = [(i_hat * k + x, j_hat * k + y) for x in range(k) for y in range(k)]
    positions_2 = [(i + x, j + y) for x in range(k) for y in range(k)]

    return matrix_1, matrix_2, positions_1, positions_2


def get_block_number_case1(n, k, i, j):
    block_row = i // k
    block_col = j // k
    blocks_per_row = (n // k) - 1
    block_number = block_row * blocks_per_row + block_col
    return block_number


def get_block_number_case2(n, k, i, j):  # And case 3
    block_row = i // k
    block_col = j // k
    blocks_per_row = (n // k)
    block_number = block_row * blocks_per_row + block_col
    return block_number


# functions for algorithm 4 #
def generate_random_matrix(n, ratio_0, ratio_1):  # Code part for testing algorithm 4
    if ratio_0 + ratio_1 != 1.0:  # Ensure the ratios sum to 1
        raise ValueError("The sum of ratio_0 and ratio_1 must be 1.0")
    total_elements = n * n  # Calculate the number of 0s and 1s
    num_0s = int(total_elements * ratio_0)
    num_1s = total_elements - num_0s
    array = np.array([0] * num_0s + [1] * num_1s)  # Create the array with the desired number of 0s and 1s
    np.random.shuffle(array)  # Shuffle the array to randomize the positions of 0s and 1s
    matrix = array.reshape((n, n))  # Reshape the array into an n x n matrix
    return matrix

def create_alternating_matrix(rows, cols):  # Code part for testing cases 2 and 3
    alternating_matrix = np.zeros((rows, cols))
    for i in range(rows):
        alternating_matrix[i] = 0 if i % 2 == 0 else 1
    return alternating_matrix


def binary_representation_block_case1(i, j, n, k):
    block_num = get_block_number_case1(n, k, i, j)
    binary_num = bin(block_num)[2:]
    num_of_bits = math.ceil(math.log2((n / k) ** 2))  # number of bits to represent the number
    num_zeros = num_of_bits - len(binary_num)  # Calculate the number of zeros needed
    binary_num = '0' * num_zeros + binary_num  # Concatenate zeros to the left
    return binary_num


def binary_representation_block_case2(i, j, n, k):
    block_num = get_block_number_case2(n, k, i, j)
    binary_num = bin(block_num)[2:]
    num_of_bits = math.ceil(math.log2((n / k) ** 2))  # number of bits to represent the number
    num_zeros = num_of_bits - len(binary_num)  # Calculate the number of zeros needed
    binary_num = '0' * num_zeros + binary_num  # Concatenate zeros to the left
    return binary_num


def generate_p_m(k):
    matrix = np.zeros((k, k), dtype=int)
    matrix[0, 0] = 1
    return matrix


def append_p_m(input_matrix, k):
    n = len(input_matrix)
    p_m = generate_p_m(k)
    new_matrix = np.zeros((n + k, n), dtype=int)  # Create the new matrix with additional space for the submatrix
    new_matrix[:n, :n] = input_matrix  # Copy the original matrix
    new_matrix[n:, :k] = p_m  # Append the submatrix
    new_matrix[n:, k:] = -1
    return new_matrix


def find_duplicate_rectangles(matrix, k, L):
    n, m = matrix.shape[0], matrix.shape[1]
    rectangles = {}
    for i in range(m - k + 1):  # Iterate over all possible positions for the rectangles
        for j in range(m - L + 1):
            rectangle = tuple(matrix[i:i + k, j:j + L].flatten())  # Flatten the blocks
            if rectangle in rectangles:  # Check if this rectangle has been seen before
                return True, rectangles[rectangle], (i, j)
            rectangles[rectangle] = (i, j)
    return False, None, None


def pad_with_ones_case2(concat_v, k):
    add = k * 3 * k - len(concat_v)
    ones = '1' * add
    concat_v += ones  # Append the random bits to concat_v
    return concat_v


def pad_with_ones_case1(concat_v, k):
    add = k * k - len(concat_v)
    ones = '1' * add
    concat_v += ones  # Append the random bits to concat_v
    return concat_v


def reshape_binary_string(binary_string, k):
    binary_list = [int(bit) for bit in binary_string]
    required_length = k * k
    if len(binary_list) < required_length:
        binary_list += [0] * (required_length - len(binary_list))
    else:
        binary_list = binary_list[:required_length]
    binary_array = np.array(binary_list)
    matrix = binary_array.reshape(k, k)
    return matrix


def case_1(matrix, p_m):
    n, m, k = matrix.shape[0], matrix.shape[1], p_m.shape[0]
    for i in range(n - 2 * k + 1):
        for j in range(m - k + 1):
            if np.array_equal(matrix[i:i + k, j:j + k], p_m):
                return True, (i, j)  # Return True and the starting index
    return False, None


def case_2(matrix, k):  # find_duplicate_submatrices
    ans = find_duplicate_rectangles(matrix, k, k)
    return ans


def case_3(matrix, k, L):
    ans = find_duplicate_rectangles(matrix, k, L)
    return ans


# algorithms 1 & 2 & 3 & 4 #
def alg1(input_matrix, L):
    n = len(input_matrix)
    lookup_square = np.zeros((L, L))
    input_matrix = np.array(input_matrix)
    input_matrix[n - 1, n - 1] = 1  # Put 1 in the lower right corner
    for i in range(L):  # create the lookup matrix
        for j in range(L):
            lookup_square[i, j] = input_matrix[n - L + i, n - L + j]
    e = np.array([0, 1])
    count_removal = 0
    while find_submatrix(input_matrix, n, L) is not None:
        count_removal += 1
        zero_submatrix_index = find_submatrix(input_matrix, n, L)
        v = np.array(zero_submatrix_index)
        target_index = v + e
        if check_overlap(n, L, v):
            count = count_overlaps(v, L, n)  # check how many overlaps
            vector = matrix_to_vector(lookup_square)
            removed_bits = vector[:count]  # extract the first "count" bits
            rotated_vector = vector[count:] + removed_bits
            matrix = vector_to_matrix(rotated_vector)
            for i in range(L):
                for j in range(L):
                    input_matrix[v[0] + i][v[1] + j] = matrix[i, j]
            lookup_square = generate_new_lookup(target_index, input_matrix, n, L)
        else:
            for i in range(L):
                for j in range(L):
                    input_matrix[v[0] + i][v[1] + j] = lookup_square[i][j]  # put the lookup instead of the zero submatrix
            lookup_square = generate_new_lookup(target_index, input_matrix, n, L)
        for i in range(L):
            for j in range(L):
                input_matrix[n - L + i, n - L + j] = lookup_square[i, j]
    print(count_removal)
    return input_matrix


def alg2(input_matrix, L):
    n = len(input_matrix)
    e = np.array([0, 1])
    input_matrix = np.array(input_matrix)
    while input_matrix[n - 1, n - 1] == 0:
        lookup = input_matrix[n - L:n, n - L:n]
        v_location = extract_location(lookup, input_matrix, n, e, L)
        v_location_submatrix = input_matrix[v_location[0]:v_location[0] + L, v_location[1]:v_location[1] + L]
        count = count_overlaps(v_location, L, n)  # check how many overlaps
        vector = matrix_to_vector(v_location_submatrix)
        removed_last_bits = vector[-count:]  # extract the last "count" bits
        rotated_vector = removed_last_bits + vector[:-count]
        matrix = vector_to_matrix(rotated_vector)
        for i in range(L):
            for j in range(L):
                input_matrix[n - L + i, n - L + j] = matrix[i, j]
                v_location_submatrix[i, j] = 0
    return input_matrix


def alg3_case1(new_matrix, i, j, k, L, t):
    n = len(new_matrix)
    new_matrix = remove_blocks_case1(new_matrix, k, t)
    i_hat = i // k
    j_hat = j // k
    matrix_1, matrix_2, positions_1, positions_2 = pre_replacement(new_matrix, i, j, k, i_hat, j_hat)
    overlapping_cells = list(set(positions_1).intersection(positions_2))  # Identify overlapping cells
    non_overlapping_cells_1 = [(x, y) for x, y in positions_1 if
                               (x, y) not in overlapping_cells]  # Identify non-overlapping cells
    non_overlapping_cells_2 = [(x, y) for x, y in positions_2 if (x, y) not in overlapping_cells]
    for (x1, y1), (x2, y2) in zip(non_overlapping_cells_1,
                                  non_overlapping_cells_2):  # Insert the values from non_overlapping_cells_1 to non_overlapping_cells_2 in the original matrix
        new_matrix[x2, y2] = new_matrix[x1, y1]
    blocks = matrix_to_blocks(new_matrix, k)
    location = get_block_number_case1(n, k, i, j)
    del blocks[location]
    zero_block = [[0, 0], [0, 0]]
    blocks.insert(0, zero_block)
    blocks_array = np.array(blocks)
    new_matrix = blocks_to_matrix(blocks_array)
    return new_matrix


def alg3_case2(new_matrix, i, j, k, L, t):
    n = len(new_matrix)
    k = int(L / 2)
    new_matrix = remove_blocks_case2(new_matrix, k, t)
    i_hat = i // k
    j_hat = j // k
    matrix_1, matrix_2, positions_1, positions_2 = pre_replacement(new_matrix, i, j, k, i_hat, j_hat)
    overlapping_cells = list(set(positions_1).intersection(positions_2))  # Identify overlapping cells
    non_overlapping_cells_1 = [(x, y) for x, y in positions_1 if
                               (x, y) not in overlapping_cells]  # Identify non-overlapping cells
    non_overlapping_cells_2 = [(x, y) for x, y in positions_2 if (x, y) not in overlapping_cells]
    for (x1, y1), (x2, y2) in zip(non_overlapping_cells_1,
                                  non_overlapping_cells_2):  # Insert the values from non_overlapping_cells_1 to non_overlapping_cells_2 in the original matrix
        new_matrix[x2, y2] = new_matrix[x1, y1]
    blocks = matrix_to_blocks(new_matrix, k)
    location = get_block_number_case2(n, k, i, j)
    del blocks[location]  # The block we need to delete when start counting from zero
    zero_block = [[0, 0], [0, 0]]
    blocks.insert(0, zero_block)
    blocks_array = np.array(blocks)
    new_matrix = blocks_to_matrix(blocks_array)
    return new_matrix

def alg3_case3(new_matrix, i, j, k, L, t):  # And case 3
    n = len(new_matrix)
    k = int(L / 2)
    new_matrix = remove_blocks_case2(new_matrix, k, t)  # Remove t blocks
    i_hat = i // k
    j_hat = j // k
    matrix_1, matrix_2, positions_1, positions_2 = pre_replacement(new_matrix, i, j, k, i_hat, j_hat)
    overlapping_cells = list(set(positions_1).intersection(positions_2))  # Identify overlapping cells
    non_overlapping_cells_1 = [(x, y) for x, y in positions_1 if
                               (x, y) not in overlapping_cells]  # Identify non-overlapping cells
    non_overlapping_cells_2 = [(x, y) for x, y in positions_2 if (x, y) not in overlapping_cells]
    for (x1, y1), (x2, y2) in zip(non_overlapping_cells_1,
                                  non_overlapping_cells_2):  # Insert the values from non_overlapping_cells_1 to non_overlapping_cells_2 in the original matrix
        new_matrix[x2, y2] = new_matrix[x1, y1]
    blocks = matrix_to_blocks(new_matrix, k)
    location = get_block_number_case2(n, k, i, j)
    del blocks[location]  # The block we need to delete when start counting from zero
    del blocks[location+1]
    zero_block = [[0, 0], [0, 0]]
    blocks.insert(0, zero_block)
    empty_block = np.full((k, k), -1)
    blocks.append(empty_block)
    new_matrix = blocks_to_matrix(blocks)
    return new_matrix

def alg4(input_matrix, L):  # We assume that n is an integer multiple of L and L/2 (=k)
    k = int(L/2)
    input_matrix = np.array(input_matrix)
    new_matrix = append_p_m(input_matrix, k)
    n, m = new_matrix.shape
    p_m = generate_p_m(k)
    case1 = case_1(new_matrix, p_m)
    case2 = case_2(new_matrix, L)
    case3 = case_3(new_matrix, k, L)
    t = 0
    while case1[0] or case2[0] or case3[0]:
        while case1[0]:
            i, j = case1[1]
            new_matrix = alg3_case1(new_matrix, i, j, k, k, t)
            binary_rep = binary_representation_block_case1(i, j, n, k)
            concat_v = "101" + binary_rep
            concat_v = pad_with_ones_case1(concat_v, k)
            concat_v_matrix = reshape_binary_string(concat_v, k)
            new_matrix[:k, :k] = concat_v_matrix
            case1 = case_1(new_matrix, p_m)
            case2 = case_2(new_matrix, L)
            case3 = case_3(new_matrix, k, L)
        while case2[0]:
            i_1, j_1 = case2[1]
            i_2, j_2 = case2[2]
            new_matrix = alg3_case2(new_matrix, i_1, j_1, L, L, t)
            binary_rep_1 = binary_representation_block_case2(i_1, j_1, m, L)
            binary_rep_2 = binary_representation_block_case2(i_2, j_2, m, L)
            concat_v = "100" + binary_rep_1 + binary_rep_2
            concat_v = pad_with_ones_case2(concat_v, k)
            concat_v_list = [int(char) for char in concat_v]
            concat_v_matrix = np.array(concat_v_list).reshape((k, 3 * k))
            zero_block = [[0, 0], [0, 0]]
            blocks = matrix_to_blocks(new_matrix, k)
            blocks.insert(0, zero_block)
            blocks.insert(0, zero_block)
            blocks.pop()
            blocks.pop()
            blocks_array = np.array(blocks)
            new_matrix = blocks_to_matrix(blocks_array)
            new_matrix[:k, :3 * k] = concat_v_matrix
            t = 1
            case1 = case_1(new_matrix, p_m)  # updating new_matrix for cases condition
            case2 = case_2(new_matrix, L)
            case3 = case_3(new_matrix, k, L)
        while case3[0]:
            i_1, j_1 = case3[1]
            i_2, j_2 = case3[2]
            new_matrix = alg3_case3(new_matrix, i_1, j_1, k, L, t)
            binary_rep_1 = binary_representation_block_case2(i_1, j_1, m, L)
            binary_rep_2 = binary_representation_block_case2(i_2, j_2, m, L)
            concat_v = "11" + binary_rep_1 + binary_rep_2
            concat_v = pad_with_ones_case2(concat_v, k)
            concat_v_list = [int(char) for char in concat_v]

            concat_v_matrix = np.array(concat_v_list).reshape((k, 3 * k))
            zero_block = [[0, 0], [0, 0]]
            blocks = matrix_to_blocks(new_matrix, k)
            blocks.insert(0, zero_block)
            blocks.insert(0, zero_block)
            blocks.pop()
            blocks.pop()
            blocks_array = np.array(blocks)
            new_matrix = blocks_to_matrix(blocks_array)
            new_matrix[:k, :3 * k] = concat_v_matrix
            t = 1
            case1 = case_1(new_matrix, p_m)  # updating new_matrix for cases condition
            case2 = case_2(new_matrix, L)
            case3 = case_3(new_matrix, k, L)
    print('case 1, case2, case3 dont exist, We have L-Squares Unique free')

    if new_matrix.size >= m ** 2:
        print(new_matrix)
        return new_matrix

    while new_matrix.size < m ** 2:
        blocks = matrix_to_blocks(new_matrix, L)
        num_blocks_to_add = m - len(blocks)
        for _ in range(num_blocks_to_add):
            new_block = np.random.randint(0, 2, size=(L, L))
            for block in blocks:
                if np.array_equal(new_block, block):
                    new_block = np.random.randint(0, 2, size=(L, L))
            blocks.append(new_block)

    return new_matrix


# Main #

input_matrix_8x8 = [
    [0, 1, 0, 0, 1, 0, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 1, 0]
]

alg4(input_matrix_8x8, 4)