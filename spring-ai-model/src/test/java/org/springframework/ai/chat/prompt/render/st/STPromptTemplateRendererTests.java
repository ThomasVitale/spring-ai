/*
 * Copyright 2023-2025 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.springframework.ai.chat.prompt.render.st;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.HashMap;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.prompt.template.st.StTemplateRenderer;
import org.springframework.test.util.ReflectionTestUtils;

/**
 * Unit tests for {@link StTemplateRenderer}.
 *
 * @author Thomas Vitale
 */
class STPromptTemplateRendererTests {

	@Test
	void shouldNotAcceptNullValidationMode() {
		assertThatThrownBy(() -> StTemplateRenderer.builder().validationMode(null).build())
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("validationMode cannot be null");
	}

	@Test
	void shouldUseDefaultValuesWhenUsingBuilder() {
		StTemplateRenderer renderer = StTemplateRenderer.builder().build();

		assertThat(ReflectionTestUtils.getField(renderer, "startDelimiterToken")).isEqualTo('{');
		assertThat(ReflectionTestUtils.getField(renderer, "endDelimiterToken")).isEqualTo('}');
		assertThat(ReflectionTestUtils.getField(renderer, "validationMode"))
			.isEqualTo(StTemplateRenderer.ValidationMode.THROW);
	}

	@Test
	void shouldRenderTemplateWithSingleVariable() {
		StTemplateRenderer renderer = StTemplateRenderer.builder().build();
		Map<String, Object> variables = new HashMap<>();
		variables.put("name", "Spring AI");

		String result = renderer.apply("Hello {name}!", variables);

		assertThat(result).isEqualTo("Hello Spring AI!");
	}

	@Test
	void shouldRenderTemplateWithMultipleVariables() {
		StTemplateRenderer renderer = StTemplateRenderer.builder().build();
		Map<String, Object> variables = new HashMap<>();
		variables.put("greeting", "Hello");
		variables.put("name", "Spring AI");
		variables.put("punctuation", "!");

		String result = renderer.apply("{greeting} {name}{punctuation}", variables);

		assertThat(result).isEqualTo("Hello Spring AI!");
	}

	@Test
	void shouldNotRenderEmptyTemplate() {
		StTemplateRenderer renderer = StTemplateRenderer.builder().build();
		Map<String, Object> variables = new HashMap<>();

		assertThatThrownBy(() -> renderer.apply("", variables)).isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("template cannot be null or empty");
	}

	@Test
	void shouldNotAcceptNullVariables() {
		StTemplateRenderer renderer = StTemplateRenderer.builder().build();
		assertThatThrownBy(() -> renderer.apply("Hello!", null)).isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("variables cannot be null");
	}

	@Test
	void shouldNotAcceptVariablesWithNullKeySet() {
		StTemplateRenderer renderer = StTemplateRenderer.builder().build();
		String template = "Hello!";
		Map<String, Object> variables = new HashMap<String, Object>();
		variables.put(null, "Spring AI");

		assertThatThrownBy(() -> renderer.apply(template, variables)).isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("variables keys cannot be null");
	}

	@Test
	void shouldThrowExceptionForInvalidTemplateSyntax() {
		StTemplateRenderer renderer = StTemplateRenderer.builder().build();
		Map<String, Object> variables = new HashMap<>();
		variables.put("name", "Spring AI");

		assertThatThrownBy(() -> renderer.apply("Hello {name!", variables)).isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("The template string is not valid.");
	}

	@Test
	void shouldThrowExceptionForMissingVariablesInThrowMode() {
		StTemplateRenderer renderer = StTemplateRenderer.builder().build();
		Map<String, Object> variables = new HashMap<>();
		variables.put("greeting", "Hello");

		assertThatThrownBy(() -> renderer.apply("{greeting} {name}!", variables))
			.isInstanceOf(IllegalStateException.class)
			.hasMessageContaining(
					"Not all variables were replaced in the template. Missing variable names are: [name]");
	}

	@Test
	void shouldContinueRenderingWithMissingVariablesInWarnMode() {
		StTemplateRenderer renderer = StTemplateRenderer.builder()
			.validationMode(StTemplateRenderer.ValidationMode.WARN)
			.build();
		Map<String, Object> variables = new HashMap<>();
		variables.put("greeting", "Hello");

		String result = renderer.apply("{greeting} {name}!", variables);

		assertThat(result).isEqualTo("Hello !");
	}

	@Test
	void shouldRenderWithoutValidationInNoneMode() {
		StTemplateRenderer renderer = StTemplateRenderer.builder()
			.validationMode(StTemplateRenderer.ValidationMode.NONE)
			.build();
		Map<String, Object> variables = new HashMap<>();
		variables.put("greeting", "Hello");

		String result = renderer.apply("{greeting} {name}!", variables);

		assertThat(result).isEqualTo("Hello !");
	}

	@Test
	void shouldRenderWithCustomDelimiters() {
		StTemplateRenderer renderer = StTemplateRenderer.builder()
			.startDelimiterToken('<')
			.endDelimiterToken('>')
			.build();
		Map<String, Object> variables = new HashMap<>();
		variables.put("name", "Spring AI");

		String result = renderer.apply("Hello <name>!", variables);

		assertThat(result).isEqualTo("Hello Spring AI!");
	}

	@Test
	void shouldHandleSpecialCharactersAsDelimiters() {
		StTemplateRenderer renderer = StTemplateRenderer.builder()
			.startDelimiterToken('$')
			.endDelimiterToken('$')
			.build();
		Map<String, Object> variables = new HashMap<>();
		variables.put("name", "Spring AI");

		String result = renderer.apply("Hello $name$!", variables);

		assertThat(result).isEqualTo("Hello Spring AI!");
	}

	@Test
	void shouldHandleComplexTemplateStructures() {
		StTemplateRenderer renderer = StTemplateRenderer.builder().build();
		Map<String, Object> variables = new HashMap<>();
		variables.put("header", "Welcome");
		variables.put("user", "Spring AI");
		variables.put("items", "one, two, three");
		variables.put("footer", "Goodbye");

		String result = renderer.apply("""
				{header}
				User: {user}
				Items: {items}
				{footer}
				""", variables);

		assertThat(result).isEqualToNormalizingNewlines("""
				Welcome
				User: Spring AI
				Items: one, two, three
				Goodbye
				""");
	}

}
