#include "Util.h"

using namespace std;

string get_bits(unsigned int x)
{
  string ret;
  for (unsigned int mask=0x80000000; mask; mask>>=1) {
    ret += (x & mask) ? "1" : "0";
  }
  return ret;
}

string get_bits(int x)
{
  string ret;
  
  for(unsigned int i = 0; i < sizeof(int) * CHAR_BIT; ++i, x >>= 1) {
	  ret += ((x & 1) == 1 ? "1" : "0");
  }

  return ret;
}

string get_bits(unsigned char x)
{
  string ret;
  
  for(unsigned int i = 0; i < sizeof(unsigned char) * CHAR_BIT; ++i, x >>= 1) {
	  ret += ((x & 1) == 1 ? "1" : "0");
  }

  return ret;
}

