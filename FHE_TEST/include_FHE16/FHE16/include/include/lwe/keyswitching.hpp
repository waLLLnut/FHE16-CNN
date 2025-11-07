#ifndef FHE16_KEYSWITCHING_CSTYLE_H
#define FHE16_KEYSWITCHING_CSTYLE_H

#include<stdint.h>
#include<FHE16Param.hpp>
void C_KeySwitchingRawCRTBin_16bit(int32_t * val, int16_t *KS_raw_16bit, int16_t *MEMORY_HANDLE, FHE16Params *PARAM, NTTTable16Struct *st, int LOC);

#endif // End header

