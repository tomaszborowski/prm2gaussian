import io
import pytest
from read_prmtop import prmtop_read_numeric_section

def test_prmtop_read_numeric_section_valid():
    file_content = """
%FLAG CHARGE
%FORMAT(5E16.8)
  0.00000000E+00  1.00800000E+00  4.00260000E+00  6.94000000E+00  9.01220000E+00
  1.08100000E+01  1.20110000E+01  1.40070000E+01  1.59990000E+01  1.89980000E+01
"""
    file = io.StringIO(file_content)
    result = prmtop_read_numeric_section(file, 'CHARGE', 10)
    expected = [0.0, 1.008, 4.0026, 6.94, 9.0122, 10.81, 12.011, 14.007, 15.999, 18.998]
    assert result == expected

def test_prmtop_read_numeric_section_invalid_flag():
    file_content = """
%FLAG CHARGE
%FORMAT(5E16.8)
  0.00000000E+00  1.00800000E+00  4.00260000E+00  6.94000000E+00  9.01220000E+00
  1.08100000E+01  1.20110000E+01  1.40070000E+01  1.59990000E+01  1.89980000E+01
"""
    file = io.StringIO(file_content)
    with pytest.raises(AssertionError, match="not a valid FLAG"):
        prmtop_read_numeric_section(file, 'INVALID_FLAG', 10)

def test_prmtop_read_numeric_section_invalid_length():
    file_content = """
%FLAG CHARGE
%FORMAT(5E16.8)
  0.00000000E+00  1.00800000E+00  4.00260000E+00  6.94000000E+00  9.01220000E+00
  1.08100000E+01  1.20110000E+01  1.40070000E+01  1.59990000E+01  1.89980000E+01
"""
    file = io.StringIO(file_content)
    with pytest.raises(AssertionError, match="length of a numeric section seems not to match the expected length inferred from pointers"):
        prmtop_read_numeric_section(file, 'CHARGE', 5)

def test_prmtop_read_numeric_section_empty_file():
    file_content = ""
    file = io.StringIO(file_content)
    with pytest.raises(AssertionError, match="length of a numeric section seems not to match the expected length inferred from pointers"):
        prmtop_read_numeric_section(file, 'CHARGE', 10)