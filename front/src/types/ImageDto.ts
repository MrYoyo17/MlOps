export interface ImageDto {
    id: string;
    imageUrl: string;
    processeds?: {
        result: {
            beard: boolean;
            mustache: boolean;
            glasses: boolean;
            hairColor: string;
            hairLength: string;
        };
    }[];
}