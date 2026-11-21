package com.faceguard.backend.image;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;

import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.http.ContentDisposition;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import com.faceguard.backend.common.ApiException;

@Tag(name = "사진")
@RestController
@RequestMapping("/api/images")
public class ImageController {

    private final ImageService service;

    public ImageController(ImageService service) {
        this.service = service;
    }

    /** 사진 올리기 → 사진 번호 */
    @Operation(summary = "사진 올리기", description = "JPEG, PNG, WEBP, 최대 20MB. 응답의 id로 보호 작업을 만든다")
    @PostMapping(consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<ImageResponse> upload(@RequestParam("file") MultipartFile file) {
        Image image = service.saveUpload(file);
        return ResponseEntity.status(HttpStatus.CREATED).body(ImageResponse.from(image));
    }

    /** 사진 정보 */
    @Operation(summary = "사진 정보")
    @GetMapping("/{id}/info")
    public ImageResponse info(@PathVariable Long id) {
        return ImageResponse.from(service.get(id));
    }

    /** 사진 파일 내려받기 (원본 또는 보호된 사진) */
    @Operation(summary = "사진 내려받기", description = "원본 또는 보호된 사진 파일")
    @GetMapping("/{id}")
    public ResponseEntity<Resource> download(@PathVariable Long id) {
        Image image = service.get(id);
        FileSystemResource file = new FileSystemResource(service.pathOf(image));
        if (!file.exists()) {
            throw new ApiException(HttpStatus.NOT_FOUND, "사진 파일이 없습니다: " + id);
        }
        String filename = image.getOriginalFilename() == null ? "image" : image.getOriginalFilename();
        return ResponseEntity.ok()
                .contentType(MediaType.parseMediaType(image.getContentType()))
                .header(HttpHeaders.CONTENT_DISPOSITION,
                        ContentDisposition.inline().filename(filename, java.nio.charset.StandardCharsets.UTF_8).build().toString())
                .body(file);
    }
}
