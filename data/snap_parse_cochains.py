"""Provides utilities to parse and process SNAP Facebook connection data"""

from typing import Dict, Set, List
from collections import defaultdict

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

    friend_dict = defaultdict(set())

    with open(file_path, 'r') as file:
        for line in file:
            user, friend = *line.split()
            friend_dict[user].add(friend)
    return friend_dict

def cochain_val(users: List[str]):
    """ Returns the number of friends shared by each user in users.

    Args:
        users: A list of users for which to check mutual friends

    Returns:
    """
    return
