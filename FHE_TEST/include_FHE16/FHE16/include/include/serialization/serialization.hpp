#ifndef FHE16_SERIALIZATON_H
#define FHE16_SERIALIZATON_H


#define _FILE_OFFSET_BITS 64

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// ------------------------------------------------------------
// 공통
// ------------------------------------------------------------
enum { FHEIO_ALIGN64 = 64 };

// ------------------------------------------------------------
// KeyPack V2 (.keys)
// [KeyPackHeaderV2 128B]
//   - 각 키 섹션(선택적 AUT 포함)은 (len, elem_bits, offset) 기록
//   - 섹션 데이터는 64B 경계 정렬로 연속 저장
//   - elem_bits: 16 또는 32 (다른 값은 오류)
// ------------------------------------------------------------
#pragma pack(push, 1)
typedef struct {
    uint64_t len;        // 요소 개수 (elements)
    uint64_t offset;     // 파일 오프셋 (64B aligned)
    uint32_t elem_bits;  // 16 or 32
    uint32_t reserved;   // 0
} KeyEntry;              // 24B
#pragma pack(pop)

#pragma pack(push, 1)
typedef struct {
    uint32_t magic;      // 'KEY2' = 0x3259454B
    uint32_t version;    // 2
    uint32_t endian;     // 1=little
    uint32_t flags;      // bit0: has_aut(ROT)

    KeyEntry brk;        // Blind/Boot Key
    KeyEntry aut;        // Automorphism/Rotation (optional)
    KeyEntry ksk;        // KeySwitching Key
    KeyEntry pk;         // Public Key

    uint64_t reserved[2];// header 크기 정확히 128B 유지
} KeyPackHeaderV2;       // 16 + (24*4) + 16 = 128B
#pragma pack(pop)

#if defined(__cplusplus)
static_assert(sizeof(KeyPackHeaderV2) == 128, "KeyPackHeaderV2 must be 128 bytes");
#else
_Static_assert(sizeof(KeyPackHeaderV2) == 128, "KeyPackHeaderV2 must be 128 bytes");
#endif

// mmap 뷰 (타입/길이 포함)
typedef struct {
    int       fd;
    size_t    filesize;
    void     *base;

    const void *brk; size_t brk_len; uint32_t brk_bits;
    const void *aut; size_t aut_len; uint32_t aut_bits;
    const void *ksk; size_t ksk_len; uint32_t ksk_bits;
    const void *pk;  size_t pk_len;  uint32_t pk_bits;
} KeyPackMapV2;

// ------------------------------------------------------------
// API
// ------------------------------------------------------------

// Generic 저장 (섹션별 포인터/길이/비트 폭 지정)
// - aut_len==0 또는 aut==NULL이면 AUT 섹션 미기록
int keypack2_save(const char *path,
                  const void *brk, uint64_t brk_len, uint32_t brk_bits,
                  const void *aut, uint64_t aut_len, uint32_t aut_bits,
                  const void *ksk, uint64_t ksk_len, uint32_t ksk_bits,
                  const void *pk,  uint64_t pk_len,  uint32_t pk_bits);

// Generic 로드 (mmap) + 검증
int  keypack2_mmap_load(const char *path, KeyPackMapV2 *out);
void keypack2_unmap(KeyPackMapV2 *m);

// 전체 섹션 CRC32C(헤더에 저장됨) 검증 (성공 0, 실패 -1/EBADMSG)
int  keypack2_crc32c_verify(const KeyPackMapV2 *km);

// ------------------------------------------------------------
// FHE16 전용 편의 래퍼 (start 포인터 사용)
//   - len 값은 호출자가 넘겨줍니다. (스키마별 크기 공식이 다르므로)
//   - brk_bits/aut_bits/ksk_bits/pk_bits는 기본 16/16/16/32 권장
// ------------------------------------------------------------
int fhe16_save_keys_from_starts(const char *path,
                                const void *BK_start,  uint64_t BK_len,  uint32_t BK_bits,
                                const void *ROT_start, uint64_t ROT_len, uint32_t ROT_bits,
                                const void *KS_start,  uint64_t KS_len,  uint32_t KS_bits,
                                const void *PK_start,  uint64_t PK_len,  uint32_t PK_bits);

// CRC32C 함수(단독 사용 가능)
uint32_t fhe_crc32c(const void *data, size_t len);

#ifdef __cplusplus
} // extern "C"
#endif



#endif // End header
