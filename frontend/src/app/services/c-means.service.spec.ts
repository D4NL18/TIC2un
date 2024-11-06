import { TestBed } from '@angular/core/testing';

import { CMeansService } from './c-means.service';

describe('CMeansService', () => {
  let service: CMeansService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(CMeansService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
