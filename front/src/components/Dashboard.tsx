import { useState } from 'react';
import { Filter, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select"
import { ImageGrid } from './ImageGrid';
import { useQueryClient } from '@tanstack/react-query';

export function Dashboard() {
    const queryClient = useQueryClient();
    const [filters, setFilters] = useState<{
        beard?: boolean;
        mustache?: boolean;
        glasses?: boolean;
        hairColor?: 'blond' | 'lightBrown' | 'red' | 'darkBrown' | 'grayBlue';
        hairLength?: 'long' | 'short' | 'bald';
        model?: string;
    }>({});

    const handleFilterChange = (key: keyof typeof filters, value: string) => {
        setFilters(prev => {
            const newFilters = { ...prev };
            if (value === 'any') {
                delete newFilters[key];
            } else if (value === 'true') {
                // @ts-ignore
                newFilters[key] = true;
            } else if (value === 'false') {
                // @ts-ignore
                newFilters[key] = false;
            } else {
                // @ts-ignore
                newFilters[key] = value;
            }
            return newFilters;
        });
    };

    const resetFilters = () => setFilters({});

    const refresh = () => {
        queryClient.invalidateQueries({ queryKey: ['images'] });
    };


    return (
        <div className="container mx-auto py-8 space-y-8">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight">MLOps Dashboard</h1>
                    <p className="text-muted-foreground">Manage images and view predictions</p>
                </div>
                <div className="flex gap-2">
                    <Button variant="outline" onClick={refresh}>
                        <RefreshCw className="mr-2 h-4 w-4" />
                        Refresh
                    </Button>
                </div>
            </div>

            {/* Content */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                <Card className="md:col-span-1 h-fit sticky top-4">
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <Filter className="h-5 w-5" />
                            Filters
                        </CardTitle>
                        <CardDescription>Refine your image search</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="space-y-2">
                            <label className="text-sm font-medium">Beard</label>
                            <Select
                                value={filters.beard === undefined ? 'any' : String(filters.beard)}
                                onValueChange={(v) => handleFilterChange('beard', v)}
                            >
                                <SelectTrigger>
                                    <SelectValue placeholder="Any" />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="any">Any</SelectItem>
                                    <SelectItem value="true">Yes</SelectItem>
                                    <SelectItem value="false">No</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>

                        <div className="space-y-2">
                            <label className="text-sm font-medium">Mustache</label>
                            <Select
                                value={filters.mustache === undefined ? 'any' : String(filters.mustache)}
                                onValueChange={(v) => handleFilterChange('mustache', v)}
                            >
                                <SelectTrigger>
                                    <SelectValue placeholder="Any" />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="any">Any</SelectItem>
                                    <SelectItem value="true">Yes</SelectItem>
                                    <SelectItem value="false">No</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>

                        <div className="space-y-2">
                            <label className="text-sm font-medium">Glasses</label>
                            <Select
                                value={filters.glasses === undefined ? 'any' : String(filters.glasses)}
                                onValueChange={(v) => handleFilterChange('glasses', v)}
                            >
                                <SelectTrigger>
                                    <SelectValue placeholder="Any" />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="any">Any</SelectItem>
                                    <SelectItem value="true">Yes</SelectItem>
                                    <SelectItem value="false">No</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>

                        <div className="space-y-2">
                            <label className="text-sm font-medium">Hair Color</label>
                            <Select
                                value={filters.hairColor || 'any'}
                                onValueChange={(v) => handleFilterChange('hairColor', v)}
                            >
                                <SelectTrigger>
                                    <SelectValue placeholder="Any" />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="any">Any</SelectItem>
                                    <SelectItem value="blond">Blond</SelectItem>
                                    <SelectItem value="lightBrown">Light Brown</SelectItem>
                                    <SelectItem value="red">Red</SelectItem>
                                    <SelectItem value="darkBrown">Dark Brown</SelectItem>
                                    <SelectItem value="grayBlue">Gray Blue</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>

                        <div className="space-y-2">
                            <label className="text-sm font-medium">Hair Length</label>
                            <Select
                                value={filters.hairLength || 'any'}
                                onValueChange={(v) => handleFilterChange('hairLength', v)}
                            >
                                <SelectTrigger>
                                    <SelectValue placeholder="Any" />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="any">Any</SelectItem>
                                    <SelectItem value="bald">Bald</SelectItem>
                                    <SelectItem value="short">Short</SelectItem>
                                    <SelectItem value="long">Long</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>

                        <Separator />

                        <Button className="w-full" variant="secondary" onClick={resetFilters}>Reset Filters</Button>
                    </CardContent>
                </Card>

                <div className="md:col-span-3 space-y-6">
                    <Card>
                        <CardHeader>
                            <CardTitle>Image Gallery</CardTitle>
                            <CardDescription>Browsing uploaded images</CardDescription>
                        </CardHeader>
                        <CardContent>
                            <ImageGrid filters={filters} />
                        </CardContent>
                    </Card>
                </div>
            </div>
        </div>
    );
}
