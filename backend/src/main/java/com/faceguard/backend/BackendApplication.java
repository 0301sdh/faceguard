package com.faceguard.backend;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import io.swagger.v3.oas.annotations.OpenAPIDefinition;
import io.swagger.v3.oas.annotations.info.Info;

@OpenAPIDefinition(info = @Info(title = "FaceGuard Backend", version = "0.1",
		description = "사진 올리기 → 보호 작업 만들기 → 보호된 사진 내려받기. AI 서버(localhost:8000)가 켜져 있어야 한다"))
@SpringBootApplication
public class BackendApplication {

	public static void main(String[] args) {
		SpringApplication.run(BackendApplication.class, args);
	}

}
