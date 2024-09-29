__author__ = "Yun Sion 33413592"
# Github = https://github.com/Sion-Yun/FIT3155_A2

import heapq
from collections import Counter
import sys

####################
# Util
####################

def dec_to_bin(k: int):
    """
    Converts a decimal number to a binary number expression
    :param k: the integer to convert
    :return binary_string: the binary expression string

    time complexity: O(log(k))
    space complexity: O(log(k))
    """
    binary_string = ""
    while k >= 1:
        if k % 2 == 0:
            binary_string += "0"
        else:
            binary_string += "1"
        k //= 2

    # Returns the reversed binary string
    return binary_string[::-1]

def elias_encode(k: int):
    """
    Converts an integer to Elias encoding
    :param k: the integer to convert
    :return elias_bin: the Elias binary expression converted

    time complexity: O(log(k)^2)
         - Each step involves binary conversion and adding length prefixes
    space complexity: O(log(k))
        - Storing the binary string
    """
    elias_bin = ""
    # convert k+1 as Elias codeword is non-negative
    k_bin = dec_to_bin(k + 1)
    elias_bin += k_bin
    x = len(k_bin) - 1

    while x >= 1:
        # zeros to the x (bin) and concat to the Elias binary
        front_bin = dec_to_bin(x)
        front_bin = "0" + front_bin[1:len(front_bin)]
        elias_bin = front_bin + elias_bin
        x = len(front_bin) - 1
    return elias_bin


class HuffmanNode:
    """
    Represents the node of a Huffman tree
    """
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char  # the character stored in the node
        self.freq = freq  # the freq of char / combined freq of children
        self.left = left  # left child node
        self.right = right  # right child node

    def __lt__(self, other):
        # Comparison for heapq based on freq
        return self.freq < other.freq

def build_huffman_tree(txt: str):
    """
    Builds the Huffman tree from a string
    :param txt: the input text
    :return: the root node of Huffman tree

    time complexity: O(n log n), n is the number of distinctive chars in the string
        - building a priority queue and merging nodes
    space complexity: O(m), m is the number of nodes in the tree

    """
    freq = Counter(txt)  # the freq of each char
    heap = [HuffmanNode(char, freq) for char, freq in freq.items()]
    heapq.heapify(heap)  # get min-heap (priority queue)

    # merging the two lowest freq nodes until only one node remains
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq, left, right)
        heapq.heappush(heap, merged)

    return heap[0]  # Return the root node

def huffman_encode(node, prefix="", huffman_codes={}):
    """
    Recursively generates the Huffman codes for each char in the Huffman tree
    :param node: (initially) the root node / the current node
    :param prefix: the current Huffman code prefix
    :param huffman_codes: dictionary to store the generated codes
    :return huffman_codes: the dictionary of Huffman codes for each char

    time complexity: O(n), n is the number of nodes in the Huffman tree.
    space complexity: O(n), n is the number of nodes in the Huffman tree.
    """
    # Start from the root node, build unique codes (paths) for each char (node)
    if node.char is not None:
        huffman_codes[node.char] = prefix
    if node.left is not None:
        huffman_encode(node.left, prefix + "0", huffman_codes)
    if node.right is not None:
        huffman_encode(node.right, prefix + "1", huffman_codes)
    return huffman_codes

def lz77_encode(txt: str):
    """
    Encode a string to LZ77 format
    :param txt: the input string to convert
    :return: the list of LZ77 tuples <offset, length, next char>

    time complexity: O(n^2), where n is len(txt), for matching substrings
    space complexity: O(n), where n is len(txt), for storing LZ77 tuples
    """
    lz77_list = []  # return list
    pos = 0  # current pos in the txt

    while pos < len(txt):
        ptr = pos - 1
        match_counts = []

        # Look for the longest match between the current pos and the window
        while ptr >= 0:
            match_count = 0
            for i in range(len(txt) - pos):
                if txt[ptr] != txt[pos]:
                    break
                if pos + i <= len(txt) - 1 and txt[ptr + i] == txt[pos + i]:
                    match_count += 1
                else:
                    break
            match_counts.append([match_count, pos - ptr])
            if match_count == len(txt) - pos:  # break if the match covers the lookahead buffer size
                break
            ptr -= 1

        # The maximum match and its offset
        max_match, offset = max(match_counts, key=lambda x: x[0]) if match_counts else (0, 0)
        next_char = txt[pos + max_match] if pos + max_match < len(txt) else ''

        # Append ⟨offset, length, next_char⟩ to the return list
        lz77_list.append([offset, max_match, next_char])

        # Move pos forward
        pos += max_match + 1  # move past the match + next_char

    # print("encoded arr from lz77_encode: ", encoded_arr)
    return lz77_list  # return array of LZ77 format

###############################
# Encoding
###############################

class Encode:
    """
    Performs the overall encoding!
    """
    def __init__(self, filename, txt):
        self.file_size = len(txt)

        # LZ77 compression on the input text
        self.lzss_tuples = lz77_encode(txt)

        # Huffman codes generated from Huffman tree
        self.huffman_codes = huffman_encode(build_huffman_tree(txt))

        # Combine header and LZ77 encoded data
        encoded_output = self.encode_header(filename) + self.encode_lz77_data()
        # print(encoded_output)

        # Write the encoded output to binary file
        output_filename = filename.split('.')[0] + ".bin"
        # output_filename = filename.split('.')[0] + ".txt"
        self.output(encoded_output, output_filename)

    def encode_header(self, filename):
        """
        Encodes the header, which contains the metadata and the huffman code info
        :param filename: the name of input/output file
        :return: the encoded header

        time complexity: O(n), where n = len(filename) + number of distinct chars in the file
        space complexity: O(n), where n = len(filename) + number of distinct chars in the file
        """
        encoded_header = ""

        # Encode file size using Elias encoding
        # print(elias_encode(self.file_size))
        encoded_header += elias_encode(self.file_size)

        # Encode filename length using Elias encoding
        filename_length = len(filename)
        # print(elias_encode(filename_length))
        encoded_header += elias_encode(filename_length)

        # Encode the filename using 8-bit ASCII
        for char in filename:
            # print(dec_to_bin(ord(char)).zfill(8))
            encoded_header += dec_to_bin(ord(char)).zfill(8)

        # Encode the number of distinct characters
        distinct_chars = len(self.huffman_codes)
        # print(elias_encode(distinct_chars))
        encoded_header += elias_encode(distinct_chars)

        # For each character, encode its ASCII, codeword length, and the Huffman code
        for char, code in self.huffman_codes.items():
            encoded_header += dec_to_bin(ord(char)).zfill(8)  # 8-bit ASCII
            encoded_header += elias_encode(len(code))  # Codeword length
            encoded_header += code  # Huffman code itself
            """
            This code gives distinct chars in non-binary order,  (e.g. b = 00, c = 01, a = 1)
            This differs from the A2 specs example but this could be encoded in any order, according to the specs
            """
            # print(char, dec_to_bin(ord(char)).zfill(8), elias_encode(len(code)), code)
        return encoded_header

    def encode_lz77_data(self):
        """
        Encodes the LZ77 tuple data parts
        :return: the encoded LZ77 tuple data

        time complexity: O(n log n), where n = len(LZ77 tuples)
        space complexity: O(n), where n = len(LZ77 tuples)
        """
        encoded_data = ""

        for lz77_tuple in self.lzss_tuples:
            # print(lz77_tuple)
            if len(lz77_tuple) == 3:  # Tuple ⟨offset, length, next_char⟩
                offset, length, next_char = lz77_tuple
                encoded_data += elias_encode(offset)  # offset using Elias
                encoded_data += elias_encode(length)  # length using Elias
                encoded_data += self.huffman_codes[next_char]  # next_char using Huffman
                # print(elias_encode(offset), elias_encode(length), self.huffman_codes[next_char])
            else:  # Tuple of form ⟨literal⟩
                literal = lz77_tuple[0]
                encoded_data += self.huffman_codes[literal]
                # print(self.huffman_codes[literal])
        return encoded_data

    def output(self, encoded_bits, output_filename):
        """
        Writes the encoded output into the binary output file
        :param encoded_bits: the encoded output in bits
        :param output_filename: the output file's name

        time complexity: O(n), where n = len(encoded_bits string)
        space complexity: O(n), where n = len(encoded_bits string)
        """
        # print(encoded_bits)
        byte_array = bytearray()
        for i in range(0, len(encoded_bits), 8):
            byte = encoded_bits[i:i + 8]
            byte_array.append(int(byte, 2))

        # Write to the output file
        with open(output_filename, "wb") as f:
            f.write(byte_array)


if __name__ == "__main__":
    # python a2q2.py <asc filename>
    file_name = sys.argv[1]
    with open(file_name, 'r') as f:
        text = f.read()

    Encode(file_name, text)

    # file_name = "x.asc"
    # text = "aacaacabcaba"