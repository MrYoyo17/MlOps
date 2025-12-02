import type { ImageDto } from './types/ImageDto';

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5002';

class ImagesApi {
    async imagesControllerFindAll(params: {
        limit?: number;
        offset?: number;
        beard?: boolean;
        mustache?: boolean;
        glasses?: boolean;
        hairColor?: string;
        hairLength?: string;
        model?: string;
    }): Promise<ImageDto[]> {
        const query = new URLSearchParams();
        if (params.limit) query.append('limit', params.limit.toString());
        if (params.offset) query.append('offset', params.offset.toString());
        if (params.beard !== undefined) query.append('beard', params.beard.toString());
        if (params.mustache !== undefined) query.append('mustache', params.mustache.toString());
        if (params.glasses !== undefined) query.append('glasses', params.glasses.toString());
        if (params.hairColor) query.append('hairColor', params.hairColor);
        if (params.hairLength) query.append('hairLength', params.hairLength);

        const response = await fetch(`${BASE_URL}/images?${query.toString()}`);
        if (!response.ok) {
            throw new Error('Failed to fetch images');
        }
        return response.json();
    }
}

class PredictionsApi {
    async predict(file: File): Promise<any> {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${BASE_URL}/predict`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error('Failed to predict');
        }
        return response.json();
    }
}

class ModelsApi {
    async health(): Promise<{ status: string; model_loaded: boolean }> {
        const response = await fetch(`${BASE_URL}/health`);
        if (!response.ok) {
            throw new Error('Health check failed');
        }
        return response.json();
    }
}

export const imagesApi = new ImagesApi();
export const predictionsApi = new PredictionsApi();
export const modelsApi = new ModelsApi();
