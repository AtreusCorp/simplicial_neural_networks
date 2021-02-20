"""Provides utilities to parse and process SNAP Facebook connection data"""

from typing import Dict, Set, List, Iterable, Tuple
from collections import defaultdict
from itertools import combinations
from sys import argv
import numpy as np


def parse_snap_edgesets(file_path: str) -> Dict[str, Set[str]]:
    """ Given an edge file from SNAPs Facebook dataset, parse it into a dict.

    Iterate through the lines of file. Create an empty dict mapping strings
    to sets of strings. For each line (of the form "i j"), place user j into
    the value of the dict at the user i.

    Args:
        file_path: a string representing the path of a SNAP Facebook edge set

    Returns:
        a dict mapping user ID's to sets containing all of their friend
        user IDs
    """
    friend_dict = defaultdict(set)

    with open(file_path, 'r') as file:
        for line in file:
            user, friend = line.split()
            friend_dict[user].add(friend)
    return friend_dict


def cochain_val(
        users: Iterable[str], friend_dict: Dict[str, Set[str]]) -> int:
    """ Returns the number of friends shared by each user in users.

    Args:
        users: An interable of users for which to check mutual friends
        friend_dict: The dictionary containing friend connections

    Returns:
        An integer representing the number of mutual friends shared
        by all users in users
    """

    if not users:
        return 0
    return len(set.intersection(*(friend_dict[user] for user in users)))


def build_simplices_cochains(
        friend_dict: Dict[str, Set[str]], top_deg: int) -> Tuple[
        List[Dict[frozenset[str], int]], List[Dict[
            frozenset[str], int]]]:
    """ Builds a simplicial complex from friend_dict.

    Constructs one list of dicts who map a frozenset of user ids (strings) to
    a unique identifier for that degree. The other list of dicts contain maps
    from user ids to cochain values.

    Args:
        friend_dict: The dictionary of friend relationships
        top_deg: The top degree simplex to build

    Returns:
        A tuple of lists of dicts. One list contains dicts which map user
        IDs to unique simplex identifiers. The other contains dicts mapping
        user IDs to cochain values.
    """

    simplicial_complex = []
    cochain_complex = []
    for deg in range(top_deg):
        simplicial_complex.append(dict())
        cochain_complex.append(dict())
        idx = 0

        for users in combinations(friend_dict.keys(), deg + 1):
            mutual_val = cochain_val(users=users, friend_dict=friend_dict)

            if mutual_val:
                simplicial_complex[deg][frozenset(users)] = idx
                cochain_complex[deg][frozenset(users)] = mutual_val
                idx += 1
        print(f'Processed simplices of degree {deg}')
    return simplicial_complex, cochain_complex


if __name__ == '__main__':

    # Generate simplicial and cochain complexes from given data
    assert len(argv)
    friend_dict = parse_snap_edgesets(file_path=argv[1])
    print(f'Parsed file {argv[1]}')
    simplices, cochains = build_simplices_cochains(
        friend_dict=friend_dict, top_deg=4)
    print('Built complex')
    np.save(
        f's2_3_collaboration_complex/snap_facebook_cochains.npy', cochains)
    np.save(
        f's2_3_collaboration_complex/snap_facebook_simplices.npy', simplices)
