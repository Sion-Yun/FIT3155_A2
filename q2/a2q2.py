__author__ = "Yun Sion"

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

    time complexity:

    space complexity:

    """
    binary_string = ""
    while k >= 1:
        if k % 2 == 0:
            binary_string += "0"
        else:
            binary_string += "1"
        k //= 2
    return binary_string[::-1]

def elias_encode(k: int):
    """
    Converts an integer to Elias encoding
    :param k: the integer to convert
    :return elias_bin: the Elias binary expression converted

    time complexity:

    space complexity:

    """
    elias_bin = ""
    # Plus 1 for the non-negativeness of Elias codeword (adjusting index to start from 1)
    k_bin = dec_to_bin(k + 1)
    elias_bin += k_bin
    x = len(k_bin) - 1

    while x >= 1:
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
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        # Comparison operators for heapq
        return self.freq < other.freq

def build_huffman_tree(txt: str):
    """
    Builds the Huffman tree
    :param txt: the input text
    :return: the root node of Huffman tree

    time complexity:

    space complexity:

    """
    freq = Counter(txt)
    heap = [HuffmanNode(char, freq) for char, freq in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq, left, right)
        heapq.heappush(heap, merged)

    return heap[0]  # Return the root node

def huffman_encode(node, prefix="", huffman_codes={}):
    """
    Recursively generates the Huffman codes
    :param node:
    :param prefix:
    :param huffman_codes:
    :return huffman_codes: the list of Huffman codes

    time complexity:

    space complexity:

    """
    # starting from the root node
    if node.char is not None:
        huffman_codes[node.char] = prefix
    if node.left is not None:
        huffman_encode(node.left, prefix + "0", huffman_codes)
    if node.right is not None:
        huffman_encode(node.right, prefix + "1", huffman_codes)
    return huffman_codes

def lz77_encode(txt: str):
    """
    Converts a string to LZ77 format
    :param txt: the input string to convert
    :return: the list of LZ77 expressions

    time complexity:

    space complexity:

    """
    lz77_list = []  # return list
    k = 0  # counter

    while k < len(txt):
        ptr = k - 1
        match_counts = []

        while ptr >= 0:
            match_count = 0
            for i in range(len(txt) - k):
                if txt[ptr] != txt[k]:
                    break
                if k + i <= len(txt) - 1 and txt[ptr + i] == txt[k + i]:
                    match_count += 1
                else:
                    break
            match_counts.append([match_count, k - ptr])
            if match_count == len(txt) - k:  # if number of matches = buffer size, break
                break
            ptr -= 1

        # The maximum match and its offset
        max_match, offset = max(match_counts, key=lambda x: x[0]) if match_counts else (0, 0)

        # The next character after the match, regardless of the match length
        next_char = txt[k + max_match] if k + max_match < len(txt) else ''

        # Append ⟨offset, length, next_char⟩
        lz77_list.append([offset, max_match, next_char])

        # Move past the match
        k += max_match + 1  # Always move past the match + next_char

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

        # LZ77 compression
        self.lzss_tuples = lz77_encode(txt)

        # Huffman coding
        self.huffman_codes = huffman_encode(build_huffman_tree(txt))

        # Combine header and LZ77 encoded data
        encoded_output = self.encode_header(filename) + self.encode_lz77_data()
        # print(encoded_output)

        # Write to binary file
        output_filename = filename.split('.')[0] + ".bin"
        # output_filename = filename.split('.')[0] + ".txt"
        self.output(encoded_output, output_filename)

    def encode_header(self, filename):
        """
        Encodes the header, which contains the metadata and the huffman code info
        :param filename: the name of input(=output) file
        :return: the encoded header

        time complexity:

        space complexity:

        """
        encoded_header = ""

        # 1. Encode file size using Elias encoding
        # print(elias_encode(self.file_size))
        encoded_header += elias_encode(self.file_size)

        # 2. Encode filename length using Elias encoding
        filename_length = len(filename)
        # print(elias_encode(filename_length))
        encoded_header += elias_encode(filename_length)

        # 3. Encode the filename using 8-bit ASCII
        for char in filename:
            # print(dec_to_bin(ord(char)).zfill(8))
            encoded_header += dec_to_bin(ord(char)).zfill(8)

        # 4. Encode the number of distinct characters
        distinct_chars = len(self.huffman_codes)
        # print(elias_encode(distinct_chars))
        encoded_header += elias_encode(distinct_chars)

        # 5. For each character, encode its ASCII, codeword length, and the Huffman code
        for char, code in self.huffman_codes.items():
            encoded_header += dec_to_bin(ord(char)).zfill(8)  # 8-bit ASCII
            encoded_header += elias_encode(len(code))  # Codeword length
            encoded_header += code  # Huffman code itself
            """
            this code gives distinct chars in non-binary order,  (e.g. b = 00, c = 01, a = 1)
            this differs from the A2 specs example but this could be encoded in any order, according to the specs
            """
            # print(char, dec_to_bin(ord(char)).zfill(8), elias_encode(len(code)), code)
        return encoded_header

    def encode_lz77_data(self):
        """
        Encodes the LZ77 tuple data parts
        :return: the encoded LZ77 tuple data

        time complexity:

        space complexity:

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

        time complexity:

        space complexity:

        """
        # print(encoded_bits)
        byte_array = bytearray()
        for i in range(0, len(encoded_bits), 8):
            byte = encoded_bits[i:i + 8]
            byte_array.append(int(byte, 2))

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