__author__ = "Yun Sion"
# Github = https://github.com/Sion-Yun/FIT3155_A2
import sys

"""
# get BWT of the input text
# search the given pattern,
# using BWT as a search index
"""

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

def get_bwt(txt: str) -> str:
    """
    :param txt:
    :return:
    """
    # suffixarray = dfs(suffix_tree)
    # bwt = ''.join(text[i-1] for i in suffixarray)
    pass


class SuffixTree:
    # gives suffix tree of T_i for prefix str[1...i] in O(n)
    # using ukkonen algo
    def __init__(self, txt):
        self.txt = txt  # input text
        self.size = len(txt)
        self.root = None
        self.i = 0  # phase
        self.j = 0  # phase
        self.GE = None
        self.active_node = None
        self.active_edge = -1
        self.remainder_count = 0
        self.pending_node = None
        self.link = None

    def create_node(self, start, end, j, is_leaf):
        node = Node(start, end, j, is_leaf)
        node.suffix_link = self.root
        return node

    def extend_suffix_tree(self, i):
        pass

    def ukkonen(self):
        pass

class Node:
    def __init__(self, start, end, j, is_leaf):
        self.children = None
        self.suffix_link = None
        self.is_leaf = is_leaf
        self.start = None
        self.end = None


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
    txt_file = open(sys.argv[1], "r")
    pat_file = open(sys.argv[2], "r")

    Wildcard(txt_file.read(), pat_file.read())
