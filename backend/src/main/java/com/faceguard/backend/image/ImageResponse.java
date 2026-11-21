package com.faceguard.backend.image;

import java.time.Instant;

public record ImageResponse(Long id, ImageKind kind, String filename, String contentType, long sizeBytes,
                            String downloadUrl, Instant createdAt) {

    public static ImageResponse from(Image image) {
        return new ImageResponse(image.getId(), image.getKind(), image.getOriginalFilename(), image.getContentType(),
                image.getSizeBytes(), "/api/images/" + image.getId(), image.getCreatedAt());
    }
}
