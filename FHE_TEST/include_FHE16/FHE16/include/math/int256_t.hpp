// int256_t.h

#ifndef INT256_T_H
#define INT256_T_H

#include <array>
#include <iostream>
#include <stdexcept>
#include <iomanip>

class int256_t {
private:
    std::array<int16_t, 16> value{};

    void add_with_carry(int16_t& a, const int16_t b, int16_t& carry);
    void sub_with_borrow(int16_t& a, const int16_t b, int16_t& borrow);

public:
    int256_t();
    explicit int256_t(int64_t init);

    int256_t operator+(const int256_t& other) const;
    int256_t operator-(const int256_t& other) const;
    int256_t operator*(const int256_t& other) const;
    int256_t operator/(const int256_t& other) const;
    int256_t operator%(const int256_t& other) const;

    int256_t operator<<(int shift) const;
    int256_t operator>>(int shift) const;

    int256_t& operator<<=(int shift);
    int256_t& operator>>=(int shift);

	int256_t operator-() const;

    bool operator==(const int256_t& other) const;
    bool operator!=(const int256_t& other) const;
    bool operator<(const int256_t& other) const;
    bool operator<=(const int256_t& other) const;
    bool operator>(const int256_t& other) const;
    bool operator>=(const int256_t& other) const;
	
	std::string to_string() const;
    friend std::ostream& operator<<(std::ostream& os, const int256_t& num);
};

#endif // INT256_T_H
