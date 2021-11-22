import math


def dot(A, B, use_tf=False):
    total = 0
    for key, val in A.items():  # val[0] = tf, val[1] = idf
        if key in B:
            tmp = val[1] * B[key][1]
            if use_tf:
                tmp *= math.log(val[0] + 1) * math.log(B[key][0] + 1)
            total += tmp
    return total


def compute_cosine_similarity(tf_idf_a, tf_idf_b, use_tf=False):
    if not tf_idf_a or not tf_idf_b:
        return 0

    return dot(tf_idf_a, tf_idf_b, use_tf) / (
        (dot(tf_idf_a, tf_idf_a, use_tf) ** 0.5)
        * (dot(tf_idf_b, tf_idf_b, use_tf) ** 0.5)
    )


def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))


def numerical_distance(v1, v2):
    if not v1 or not v2 or v1 <= 0 or v2 <= 0:
        return 0

    if v1 < v2:
        return v1 / v2
    else:
        return v2 / v1
