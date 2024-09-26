__author__ = "Yun Sion"
# Github = https://github.com/Sion-Yun/FIT3155_A2
import sys

"""
# get BWT of the input text
# search the given pattern,
# using BWT as a search index
"""

# TODO - TEST, optimise

def z_algo(txt: str) -> [int]:
    """
    Z-algorithm; computes Z-values of a given string.

    time complexity:
        O(n), for n being the length of text.
    space complexity:
        O(n), for n being the length of text.

    :argument: txt (str): The text to find z-values.
    :return: z_arr: array of all z-values.
    """
    n = len(txt)  # input string
    z = [0] * n  # array to store Z-values
    l, r, k = 0, 0, 0  # left boundary, right boundary, and position of Z-box

    """
    Computing the Z-values
        - The set of values Z_i
        - Z_i = the length of the longest substring, starting at [i] of string, that matches its prefix.    
    """
    for i in range(1, n):
        # Case 1: k is outside the rightmost Z-box
        if i > r:
            l, r = i, i
            while r < n and txt[r - l] == txt[r]:  # explicit comparison
                r += 1
            z[i] = r - l
            r -= 1

        # Case 2: k is inside the rightmost Z-box
        else:
            # Case 2a: Z_k-l+1 box does not extend to the end of the prefix that matches Z_l box
            k = i - l
            # Case 2a
            if z[k] < r - i + 1:
                z[i] = z[k]

            # Case 2b: Z_k-l+1 box extends over the prefix that matches Z_l box
            else:
                l = i
                while r < n and txt[r - l] == txt[r]:
                    r += 1
                z[i] = r - l
                r -= 1
    return z  # return all z-values

def custom_ord(k):
    """
    Reduces ASCII value by 32 to map the chars for reduced list size
    """
    return ord(k) - 32

class GlobalEnd:
    """
    Handles the global end pointer for leaf nodes in the suffix tree, allowing dynamic extension of leaves
    """
    def __init__(self):
        self.value = None

    def set_end(self, end):
        self.value = end

    def get_end(self):
        return self.value

class Node(object):
    """
    Represents a node in the suffix tree
    """
    def __init__(self, start, end, j, is_leaf):
        # Trick 1 - space-efficient form of edge-labels and substrings
        self.children = None
        self.children_count = 0  # trick 1
        self.suffix_link = None
        self.j = j  # trick 1; If it's a leaf, this is the index of the suffix
        self.start = start  # trick 1
        self.end = end  # trick 1
        self.is_leaf = is_leaf

        if not is_leaf:
            self.set_not_leaf()

    def get_children(self, i):
        """
        Returns the child node at pos i
        """
        return self.children[i]

    def add_child(self, i, child):
        """
        Adds a child node to this node's children at pos i
        """
        self.children[i] = child
        self.children_count += 1

    def set_not_leaf(self):
        """
        Converts the node to a non-leaf node
        """
        self.children = [None] * 96
        self.is_leaf = False
        self.j = None

    def get_edge_len(self):
        """
        Returns the length of the edge represented by this node.
        """
        return self.get_end() - self.start + 1

    def get_end(self):
        """
        Returns the end index of the node, which can be dynamic if it's a leaf.
        """
        if isinstance(self.end, int):
            return self.end
        else:
            return self.end.get_end()

class SuffixTree(object):
    """
    A class to represent the suffix tree of a given text, using Ukkonen's algorithm for construction.
    """
    def __init__(self, txt):
        self.txt = txt
        self.n = len(txt)
        self.root = None
        self.j = 0  # current phase index of Ukkonen
        self.endPointer = GlobalEnd()  # Global end pointer for leaf nodes
        self.active_node = None
        self.prev_node = None  # previous node to maintain suffix links between internal nodes
        self.active_edge = -1
        self.active_len = 0

        self.init_suffix_tree()

    def init_suffix_tree(self):
        """
        Initialises the suffix tree by creating the root node and iterating over the text to extend.

        Time complexity: O(n), where n is the length of the input text.
        Space complexity: O(n), where n is the length of the input text.
        """
        self.root = self.create_node(-1, -1, None, False)  # root node with start and end as -1
        self.root.suffix_link = self.root  # suffix link points to itself (root)
        self.active_node = self.root  # the first active node is the root

        for i in range(self.n):
            self.extend_suffix_tree(i)  # extending each char from the text.

    def create_node(self, start, end, j, is_leaf=True):
        """
        Creates a new node of the suffix tree.
        :param start: The starting index of the node's edge in the input text.
        :param end: The ending index of the node's edge (or a reference to endPointer for leaves).
        :param j: The suffix index for leaves, None for internal nodes.
        :param is_leaf: Boolean flag to indicate if the node is a leaf.
        :return: A new node.
        """
        node = Node(start, end, j, is_leaf)
        node.suffix_link = self.root
        return node

    def extend_suffix_tree(self, i):
        """
        Extends the suffix tree for the text[i].
        Implements Ukkonen's algorithm with three extension rules.

        Time complexity: O(n), where n is the length of the input text.
            - O(1) amortized per phase.
        Space complexity: O(n), where n is the length of the input text.
        """
        # Rule 1
        # Trick 4 - rapid leaf extension
        self.endPointer.set_end(i)  # updating the global end for leaves
        self.prev_node = None  # reset the previous node for a new phase

        while self.j <= i:
            if self.active_len == 0:
                self.active_edge = i  # reset the active edge to current index

            # Rule 2 - Branch a new leaf from the active node if no outgoing edge exists
            if self.active_node.get_children(custom_ord(self.txt[self.active_edge])) is None:
                # Create a new leaf node
                self.active_node.add_child(custom_ord(self.txt[self.active_edge]), self.create_node(i, self.endPointer, self.j))

                # If there was an internal node created in the last extension, set its suffix link
                if self.prev_node is not None:
                    self.prev_node.suffix_link = self.active_node
                    self.prev_node = None

            else:
                # Handle case where there is an outgoing edge for the active edge
                next_node = self.active_node.get_children(custom_ord(self.txt[self.active_edge]))
                next_len = next_node.get_edge_len()

                # Traverse down the tree if the active next_len exceeds the edge next_len
                if self.active_len >= next_len:
                    self.active_node = next_node
                    self.active_edge += next_len
                    self.active_len -= next_len
                    continue

                # Rule 3 - Stop extension if the character already exists on the edge
                if self.txt[i] == self.txt[next_node.start + self.active_len]:
                    if self.active_node is not self.root and self.prev_node is not None:
                        self.prev_node.suffix_link = self.active_node
                        self.prev_node = None

                    self.active_len += 1  # Extend the active next_len
                    break  # Showstopper trick

                # Rule 2 - Split the edge and create a new internal node
                new_start = next_node.start
                next_node.start += self.active_len

                # New internal node
                new_node = self.create_node(new_start, new_start + self.active_len - 1, None, False)
                new_node.add_child(custom_ord(self.txt[next_node.start]), next_node)  # close the split part
                new_node.add_child(custom_ord(self.txt[i]), self.create_node(i, self.endPointer, self.j))  # new leaf
                # active_node connect to this new internal node
                self.active_node.children[custom_ord(self.txt[new_start])] = new_node

                # Update suffix link for the previously created internal node
                if self.prev_node is not None:
                    self.prev_node.suffix_link = new_node
                self.prev_node = new_node

            self.j += 1

            # Update the active point for the next extension
            if self.active_node is self.root and self.active_len > 0:
                self.active_len -= 1
                self.active_edge = self.j
            else:
                # Follow the suffix link
                self.active_node = self.active_node.suffix_link

class BWT:
    """
    Class representing BWT of a given text.
    Uses a suffix tree for efficient suffix array construction.
    """
    def __init__(self, txt):
        self.txt = txt
        self.suffix_tree = SuffixTree(txt)
        self.suffix_array = []

        # DFS on the suffix tree to get suffix array
        self.dfs(self.suffix_tree.root)

    def get_bwt(self):
        """
        Returns the BWT string from the suffix array
        :return ret: The BWT form of the suffix array

        Time complexity: O(n), where n is the length of the input text.
        Space complexity: O(n), where n is the length of the input text.
        """
        # Convert the suffix array to the BWT string
        ret = ""
        test = ""
        for i in range(len(self.suffix_array)):
            ret += self.txt[self.suffix_array[i] - 1]
            test += self.txt[self.suffix_array[i]]
        print(self.suffix_array)  # TODO - remove after test
        return ret

    def dfs(self, node):
        """
        Depth-first search (DFS) to traverse the suffix tree and build the suffix array.
        :param node: The current node being traversed.

        Time complexity: O(n), where n is the length of the input text.
        Space complexity: O(n), where n is the length of the input text.
        """
        for i in range(96):  # 96 possible characters (ASCII values reduced by 32)
            child = node.get_children(i)
            if child is not None and child.is_leaf:
                self.suffix_array.append(child.j)  # Add leaf node's suffix index
            elif child is not None:
                self.dfs(child)  # Recurse for non-leaf nodes


class Wildcard:
    """
    A class to perform pattern matching using wildcard and Z-algorithm in segments.

    :attributes:
        total_length (int): Cumulative length of matched segments.
        segment_length (int): Length of the current segment being processed.
        currZ (list): Z-values for the current segment.
        prevZ (list): Z-values for the previous segment.
        segments (list): List of segments to be matched.
        if_no_hit (bool): Flag indicating if no match was found.
        n (int): Length of the input text.
        txt (str): The text to be matched against.
        wild_length (int): Length of the wildcard segment (when applicable).
    """
    def __init__(self, txt, pat):
        """
        The initialisation of Wildcard class.

        :param txt: the text to match.
        :param pat: the pattern to match.
        """
        self.txt = txt
        self.pat = pat
        self.n = len(txt)
        self.currZ = []
        self.prevZ = []
        self.segments = self.extract_substrings()
        self.total_length = 0
        self.segment_length = 0
        self.wild_length = 0
        self.is_no_hit = False

        self.match()
        self.output()

    def extract_substrings(self):
        """
        Extracts substrings from the pattern, separated by the '!' character.

        This method iterates through the input pattern, using '!' as delimiter.
        The substrings are added to a list, which is the return.

        time complexity:
            O(m), for m being the length of pat.
        space complexity:
            O(m), for m being the length of pat.

        :param self:
        :return: A list of substrings extracted from pat.txt that are separated by '!'.
        """
        substrings = []  # array of the substrings
        sub = ""  # the substring
        # k = 0  # counter
        m = len(self.pat)

        for i in range(m):
            if self.pat[i] == '!':
                # a wildcard found
                if sub:
                    substrings.append(sub)
                    sub = ""
                substrings.append(-1)  # store -1 to represent the wildcard segment
            else:
                sub += self.pat[i]  # continue building the substring

        if sub:
            substrings.append(sub)  # append any remaining substring
        return substrings

    def merge_substrings(self):
        """
        Merges the extracted substrings (characters) with the Z-values of new segments.

        time complexity:
            O(n), for n being the length of text.
        space complexity:
            O(n), for n being the length of text.
        :param self:
        """
        arr = [0] * self.n
        k = self.total_length + self.segment_length  # the maximum length matched after the merge
        flag = False

        for i in range(self.n):
            if i + k > self.n:  # break if the remaining chars are fewer than the merged length
                break

            if self.prevZ[i] == self.total_length:
                if self.currZ[i + self.total_length + self.segment_length + 1] == self.segment_length:
                    arr[i] = k
                    flag = True

        # if no match found, set the flag (attribute) to stop further comparisons
        if not flag:
            self.is_no_hit = True

        # update the newly merged Z-values
        self.prevZ = arr

    def merge_wildcard(self):
        """
        Merges the extracted substrings with the length of new wildcard Z-segments.
        This method is used when the segment is a wildcard, which is represented as a negative value.

        time complexity:
            O(n), for n being the length of text.
        space complexity:
            O(n), for n being the length of text.
        :param self:
        """
        arr = [0] * self.n
        k = self.total_length - self.wild_length  # the maximum length matched after the merge

        # CASE: the first segment is a character segment and the array is larger than n
        shift = 0
        if len(self.prevZ) > self.n:
            shift = len(self.prevZ) - self.n

        for i in range(shift, self.n):
            if i - shift + k > self.n:  # break if the remaining characters are fewer than the merged length
                break

            if self.prevZ[i] == self.total_length:  # update the matched length in the merged array
                arr[i - shift] = k

        # update the newly merged Z - values
        self.prevZ = arr

    def match(self):
        """
        Performs the pattern matching on the text based on the segments array.
        The matching process combines results of all segments.

        time complexity:
            O(k(n + m/k)), for k: the number of segments, n: the length of the txt, m: the length of the pat.txt.
        space complexity:
            O(n), n being the length of the txt.

        :param self:
        """
        for i in range(len(self.segments)):
            if self.is_no_hit:  # stop matching further segments if no match was found in prev
                break

            if isinstance(self.segments[i], str):  # segment is a series of char, string
                merged_str = self.segments[i] + "$" + self.txt
                self.segment_length = len(self.segments[i])

                if i != 0:  # update z-values for current segment and merge with prev
                    # TODO
                    self.currZ = z_algo(merged_str)
                    self.merge_substrings()
                else:
                    # TODO
                    z_values = z_algo(merged_str)
                    self.prevZ = z_values[len(self.segments[i]) + 1:]  # skip pattern and $

                self.total_length += self.segment_length  # update the total length of matched segments
            else:
                if i == 0:  # Wildcard is the first segment
                    # Initialize prevZ to allow any position to match the next segment
                    self.prevZ = [0] * self.n
                else:
                    self.wild_length = self.segments[i]
                    self.merge_wildcard()

                # update total length
                self.total_length += -self.segments[i]

    def output(self):
        """
        Generates the output file.

        time complexity:
            O(n - m), n is the length of txt and m is the length of pat.
        """
        m = len(self.pat)
        with open("output_a1q2.txt", "w+") as f:
            if not self.is_no_hit:
                for i in range(len(self.prevZ) - m + 1):
                    if self.prevZ[i] == m:
                        f.write("%d\n" % (i + 1))


if __name__ == '__main__':
    # python a2q2.py <text filename> <pattern filename>
    # txt_file = open(sys.argv[1], "r")
    # pat_file = open(sys.argv[2], "r")

    # Wildcard(txt_file.read(), pat_file.read())
    BWT("abbbbcbbcbcabbbb").get_bwt()
    BWT("abcbaabaab").get_bwt()
    BWT("woolowwmooloo").get_bwt()
    BWT("abcab").get_bwt()
