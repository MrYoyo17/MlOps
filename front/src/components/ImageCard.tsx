import { Badge } from '@/components/ui/badge';
import { Card } from '@/components/ui/card';

import type { ImageDto } from '@/types/ImageDto';


export function ImageCard({ image }: { image: ImageDto }) {
    const prediction = image.processeds?.[0]?.result;

    return (
        <Card className="overflow-hidden py-0 h-fit transition-shadow hover:shadow-lg">
            <div className="relative aspect-square">
                <img
                    src={`${import.meta.env.VITE_API_URL}${image.imageUrl}`}
                    alt="Uploaded"
                    className="object-cover w-full h-full [image-rendering:pixelated]"
                    loading="lazy"
                />
            </div>
            {prediction && (
                <div className="p-2 border-t bg-card">
                    <div className="flex flex-wrap gap-1">
                        {prediction.beard && <Badge variant="secondary" className="text-[10px] h-4 px-1">Beard</Badge>}
                        {prediction.mustache && <Badge variant="secondary" className="text-[10px] h-4 px-1">Mustache</Badge>}
                        {prediction.glasses && <Badge variant="secondary" className="text-[10px] h-4 px-1">Glasses</Badge>}
                        <Badge variant="outline" className="text-[10px] h-4 px-1 text-muted-foreground border-border">{prediction.hairColor}</Badge>
                        <Badge variant="outline" className="text-[10px] h-4 px-1 text-muted-foreground border-border">{prediction.hairLength}</Badge>
                    </div>
                </div>
            )}
        </Card>
    );
}
