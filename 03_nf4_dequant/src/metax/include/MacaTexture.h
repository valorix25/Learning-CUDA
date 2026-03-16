#pragma once


#include "helper_maca.h"
#include <mc_runtime.h>


struct MacaTexture {
    MacaTextureObject_t tex;

    MacaTexture(MacaTexture const &) = delete;
    MacaTexture(MacaTexture &&) = default;
    MacaTexture &operator=(MacaTexture const &) = delete;
    MacaTexture &operator=(MacaTexture &&) = default;

    template <class T>
    MacaTexture(T *dataDev, int width, int height) {
        MacaTextureObject_t tex;
        mcResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = mcResourceTypePitch2D;
        resDesc.res.pitch2D.devPtr = dataDev;
        resDesc.res.pitch2D.width = width;
        resDesc.res.pitch2D.height = height;
        resDesc.res.pitch2D.desc = mcCreateChannelDesc<T>();
        resDesc.res.pitch2D.pitchInBytes = width * sizeof(T);
        MacaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        checkMacaErrors(mcCreateTextureObject(&tex, &resDesc, &texDesc, NULL));
    }

    ~MacaTexture() {
        checkMacaErrors(mcDestroyTextureObject(tex));
    }

    constexpr operator MacaTextureObject_t() const {
        return tex;
    }
};
