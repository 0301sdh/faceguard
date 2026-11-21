package com.faceguard.backend.image;

import java.time.Instant;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;

/** 저장된 사진 한 장의 기록. 파일 자체는 storage 폴더에 있고, DB에는 위치와 정보만 둔다. */
@Entity
public class Image {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private ImageKind kind;

    private String originalFilename;

    @Column(nullable = false)
    private String contentType;

    private long sizeBytes;

    @Column(nullable = false)
    private String storedName;      // storage 폴더 안의 파일 이름 (UUID)

    private Instant createdAt;

    protected Image() {
    }

    public Image(ImageKind kind, String originalFilename, String contentType, long sizeBytes, String storedName) {
        this.kind = kind;
        this.originalFilename = originalFilename;
        this.contentType = contentType;
        this.sizeBytes = sizeBytes;
        this.storedName = storedName;
        this.createdAt = Instant.now();
    }

    public Long getId() { return id; }
    public ImageKind getKind() { return kind; }
    public String getOriginalFilename() { return originalFilename; }
    public String getContentType() { return contentType; }
    public long getSizeBytes() { return sizeBytes; }
    public String getStoredName() { return storedName; }
    public Instant getCreatedAt() { return createdAt; }
}
