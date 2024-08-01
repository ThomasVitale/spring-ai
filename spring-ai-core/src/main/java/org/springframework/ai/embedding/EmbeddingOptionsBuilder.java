package org.springframework.ai.embedding;

/**
 * @author Thomas Vitale
 * @since 1.0.0
 */
public class EmbeddingOptionsBuilder {

	private final DefaultEmbeddingOptions embeddingOptions = new DefaultEmbeddingOptions();

	private EmbeddingOptionsBuilder() {
	}

	public static EmbeddingOptionsBuilder builder() {
		return new EmbeddingOptionsBuilder();
	}

	public EmbeddingOptionsBuilder withModel(String model) {
		embeddingOptions.setModel(model);
		return this;
	}

	public EmbeddingOptionsBuilder withDimensions(Integer dimensions) {
		embeddingOptions.setDimensions(dimensions);
		return this;
	}

	public EmbeddingOptions build() {
		return embeddingOptions;
	}

	private static class DefaultEmbeddingOptions implements EmbeddingOptions {

		private String model;

		private Integer dimensions;

		@Override
		public String getModel() {
			return this.model;
		}

		public void setModel(String model) {
			this.model = model;
		}

		@Override
		public Integer getDimensions() {
			return this.dimensions;
		}

		public void setDimensions(Integer dimensions) {
			this.dimensions = dimensions;
		}

	}

}
