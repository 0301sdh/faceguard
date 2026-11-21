package com.faceguard.backend.image;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.UUID;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import com.faceguard.backend.common.ApiException;

@Service
public class ImageService {

    private final ImageRepository repository;
    private final Path storageDir;

    public ImageService(ImageRepository repository, @Value("${faceguard.storage-dir}") String storageDir) {
        this.repository = repository;
        this.storageDir = Path.of(storageDir);
    }

    /** 사용자가 올린 사진을 검사하고 저장한다. */
    public Image saveUpload(MultipartFile file) {
        if (file.isEmpty()) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "빈 파일입니다");
        }
        byte[] data;
        try {
            data = file.getBytes();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        // 확장자나 사용자가 보낸 형식 정보는 바꿔 보낼 수 있으므로, 파일 앞부분(매직 넘버)으로 진짜 형식을 확인
        String contentType = detectImageType(data);
        if (contentType == null) {
            throw new ApiException(HttpStatus.UNSUPPORTED_MEDIA_TYPE, "JPEG, PNG, WEBP 사진만 올릴 수 있습니다");
        }
        return save(ImageKind.ORIGINAL, file.getOriginalFilename(), contentType, data);
    }

    /** AI 서버가 돌려준 보호된 사진(PNG)을 저장한다. */
    public Image saveProtected(Image original, byte[] png) {
        String name = original.getOriginalFilename() == null ? "photo" : original.getOriginalFilename();
        int dot = name.lastIndexOf('.');
        String base = dot > 0 ? name.substring(0, dot) : name;
        return save(ImageKind.PROTECTED, base + "_protected.png", "image/png", png);
    }

    public Image get(Long id) {
        return repository.findById(id)
                .orElseThrow(() -> new ApiException(HttpStatus.NOT_FOUND, "사진을 찾을 수 없습니다: " + id));
    }

    public Path pathOf(Image image) {
        return storageDir.resolve(image.getStoredName());
    }

    private Image save(ImageKind kind, String filename, String contentType, byte[] data) {
        String ext = switch (contentType) {
            case "image/png" -> ".png";
            case "image/webp" -> ".webp";
            default -> ".jpg";
        };
        String storedName = UUID.randomUUID() + ext;   // 사용자가 보낸 파일 이름은 경로로 쓰지 않음 (보안)
        try {
            Files.createDirectories(storageDir);
            Files.write(storageDir.resolve(storedName), data);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return repository.save(new Image(kind, filename, contentType, data.length, storedName));
    }

    static String detectImageType(byte[] d) {
        if (d.length >= 3 && (d[0] & 0xFF) == 0xFF && (d[1] & 0xFF) == 0xD8 && (d[2] & 0xFF) == 0xFF) {
            return "image/jpeg";
        }
        if (d.length >= 8 && (d[0] & 0xFF) == 0x89 && d[1] == 'P' && d[2] == 'N' && d[3] == 'G') {
            return "image/png";
        }
        if (d.length >= 12 && d[0] == 'R' && d[1] == 'I' && d[2] == 'F' && d[3] == 'F'
                && d[8] == 'W' && d[9] == 'E' && d[10] == 'B' && d[11] == 'P') {
            return "image/webp";
        }
        return null;
    }
}
